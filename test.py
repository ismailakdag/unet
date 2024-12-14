import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from os import listdir
from os.path import splitext
from pathlib import Path
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import cv2
from numpy import asarray


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits





class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray

        return img_ndarray






def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.8):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()




new_json = {"T2W_Lesion":{}}
        
full_predictions_zones = {"ImageID":[], "Segmentation":[], "Rapor":['T2W Görsellerde Tranziyonel Bölgede PIRADS-1 Lezyon Bulunmaktadır']}
        
classes_zones = {'T2W TZ PIRADS-1'}   
     
if __name__ == '__main__':
 
    
    imagePath = r'C:\Users\caglar.gurkan\Desktop\deneme1'
    
    test_imgs_paths_zones = sorted(os.listdir(imagePath + '/t2/'))
    
    for test_img_path_zones in test_imgs_paths_zones:
        
        in_files = cv2.imread(imagePath + '/t2/' + test_img_path_zones)


        
        in_files = cv2.cvtColor(in_files, cv2.COLOR_BGR2GRAY)
        
        in_files = asarray(in_files)
        
        in_files = in_files[128:384, 128:384]
        
        
        mean = np.mean(in_files)
        std = np.std(in_files)
        # Scaling
        in_files = (in_files - mean + 0.000001) / (std  + 0.000001)
        
        
        
        
        model = r'D:\Prostate_U-Nets\T2_tranzisyonel_1\unet\checkpoints\checkpoint_epoch50.pth'
    
        net = UNet(n_channels=1, n_classes=2, bilinear=False)
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
        net.to(device=device)
        net.load_state_dict(torch.load(model, map_location=device))
    
        logging.info('Model loaded!')
    

            
            
        mask = predict_img(net=net,
                           full_img=in_files,
                           scale_factor=1,
                           out_threshold=0.8,
                           device=device)
        
        mask = mask*255
        
        result = np.pad(mask[1,:,:], [128, 128], mode='constant')
        
        
        result = result.astype(np.uint8) 
        
        cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        cnts = cnts[0] if len(cnts) == 2 else cnts[1] 
        
        total_cnts = len(cnts)
        

        r=[]
        masks_zone_new=[]

        for i in range (0,total_cnts):
            
              masks_zone_new_1=[]

              for z in cnts[i]:
                  a=z[0][0].tolist()
                  b=z[0][1].tolist()
                  masks_zone_new_1.append(a)
                  masks_zone_new_1.append(b)
                
              masks_zone_new.append(masks_zone_new_1)
              
        r.append(masks_zone_new)   
              
              
        full_predictions_zones['Segmentation'].extend(r)
              
              
              
        
        test_img_path_zones_ıd = {test_img_path_zones}
        full_predictions_zones['ImageID'].extend(test_img_path_zones_ıd)
              
              
              
              
        new_json['T2W_Lesion'].update(full_predictions_zones)

