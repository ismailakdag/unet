import time ############# Burası
import argparse
import logging
import sys
from pathlib import Path


import wandb
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split



from unet import UNet


from utils.data_loading import BasicDataset, CarvanaDataset


from utils.dice_score import dice_loss
from utils.iou_score import iou_loss
from utils.tversky_score import tversky_loss
from utils.focal_tversky_score import focal_tversky_loss


from evaluate_dice import evaluate as evaluate_dice
from evaluate_iou import evaluate as evaluate_iou
from evaluate_tversky import evaluate as evaluate_tversky
from evaluate_focal_tversky import evaluate as evaluate_focal_tversky


from utils.iou import eval_net_loader as iou
from utils.specificity  import eval_net_loader as specificity
from utils.sensitivity  import eval_net_loader as sensitivity
from utils.precision  import eval_net_loader as precision
from utils.accuracy  import eval_net_loader as accuracy
from utils.dice  import eval_net_loader as dice





dir_img = Path(r'images')
dir_mask = Path(r'masks')
dir_checkpoint = Path(r'checkpoints')







def validate_epoch_iou(epoch,train_loader,val_loader,device):
                                                    
    class_iou, mean_iou = iou(net, val_loader, 2, device)
    print('Class IoU:', ' '.join(f'{x:.3f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.3f}') 
    # save to summary
    # writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))
                                                    
    return mean_iou


def validate_epoch_specificity(epoch,train_loader,val_loader,device):
                                                    
    class_Specificity, mean_Specificity = specificity(net, val_loader, 2, device)
    print('Class Specificity:', ' '.join(f'{x:.3f}' for x in class_Specificity), f'  |  Mean Specificity: {mean_Specificity:.3f}') 
    # save to summary
    # writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))
                                                    
    return mean_Specificity


def validate_epoch_sensitivity(epoch,train_loader,val_loader,device):
                                                    
    class_Sensitivity, mean_Sensitivity = sensitivity(net, val_loader, 2, device)
    print('Class Sensitivity:', ' '.join(f'{x:.3f}' for x in class_Sensitivity), f'  |  Mean Sensitivity: {mean_Sensitivity:.3f}') 
    # save to summary
    # writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))
                                                    
    return mean_Sensitivity


def validate_epoch_precision(epoch,train_loader,val_loader,device):
                                                    
    class_Precision, mean_Precision = precision(net, val_loader, 2, device)
    print('Class Precision:', ' '.join(f'{x:.3f}' for x in class_Precision), f'  |  Mean Precision: {mean_Precision:.3f}') 
    # save to summary
    # writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))
                                                    
    return mean_Precision


def validate_epoch_accuracy(epoch,train_loader,val_loader,device):
                                                    
    class_Accuracy, mean_Accuracy = accuracy(net, val_loader, 2, device)
    print('Class Accuracy:', ' '.join(f'{x:.3f}' for x in class_Accuracy), f'  |  Mean Accuracy: {mean_Accuracy:.3f}') 
    # save to summary
    # writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))
                                                    
    return mean_Accuracy


def validate_epoch_dice(epoch,train_loader,val_loader,device):
                                                    
    class_Dice, mean_Dice = dice(net, val_loader, 2, device)
    print('Class Dice:', ' '.join(f'{x:.3f}' for x in class_Dice), f'  |  Mean Dice: {mean_Dice:.3f}') 
    # save to summary
    # writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch+1))
                                                    
    return mean_Dice











def train_net(net,
              device,
              epochs: int = 100,
              batch_size: int = 2,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = True):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=(epochs*0.1))  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())




                        val_score = evaluate_dice(net, val_loader, device)
                        scheduler.step(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score))


                                                
                        iou = validate_epoch_iou(epoch, train_loader, val_loader, device)
                        # scheduler.step(iou)
                        # logging.info('Validation IoU score: {}'.format(iou))
                        
                        
                                 
                        specificity = validate_epoch_specificity(epoch, train_loader, val_loader, device)
                        # scheduler.step(specificity)
                        # logging.info('Validation Specificity score: {}'.format(specificity))
                        
                        
                        
                        sensitivity = validate_epoch_sensitivity(epoch, train_loader, val_loader, device)
                        # scheduler.step(sensitivity)
                        # logging.info('Validation Sensitivity score: {}'.format(sensitivity))
                        
                        
                        
                        precision = validate_epoch_precision(epoch, train_loader, val_loader, device)
                        # scheduler.step(precision)
                        # logging.info('Validation Precision score: {}'.format(precision))
                        
                        
                        
                        accuracy = validate_epoch_accuracy(epoch, train_loader, val_loader, device)
                        # scheduler.step(accuracy)
                        # logging.info('Validation Accuracy score: {}'.format(accuracy))
                        
                        
                        
                        dice = validate_epoch_dice(epoch, train_loader, val_loader, device)
                        # scheduler.step(dice)
                        # logging.info('Validation Dice-2 score: {}'.format(dice))
                        
                        
                        
                        
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            
                            
                            # 'validation IoU': iou,
                            
                            # 'validation Specificity': specificity,
                                           
                            # 'validation Sensitivity': sensitivity,     
                            
                            # 'validation Precision': precision,  
                            
                            # 'validation Accuracy': accuracy,
                            
                            # 'validation Dice-2': dice,
                            
                            
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2
                        , help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


if __name__ == '__main__':
    
    print("Start...") ############ Burası
    start = time.time() ############# Burası
    
    
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    
    from torchsummary import summary

    summary(net, (1, 224, 224))



    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)

end = time.time() ############# Burası
print(end - start) ############# Burası