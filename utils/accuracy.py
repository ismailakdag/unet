import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_Accuracy(cm):
    '''
    Adapted from:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
    '''
    if cm.size == 4:
        
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        
        Accuracy = (TP[1]+TN[1])/(TP[1]+FP[1]+FN[1]+TN[1])
        
        return [Accuracy], np.nanmean([Accuracy]) 
    
    else:
        
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        
        Accuracy = (TP+TN)/(TP+FP+FN+TN)
        
        return Accuracy, np.nanmean(Accuracy) 

def eval_net_loader(net, val_loader, n_classes, device='cpu'):
    
    net.eval()
    labels = np.arange(n_classes)
    cm = np.zeros((n_classes,n_classes))
      
    for i, sample_batch in enumerate(val_loader):
            imgs = sample_batch['image']
            true_masks = sample_batch['mask']
            
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            outputs = net(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            for j in range(len(true_masks)): 
                true = true_masks[j].cpu().detach().numpy().flatten()
                pred = preds[j].cpu().detach().numpy().flatten()
                cm += confusion_matrix(true, pred, labels=labels)
                
    
    
    # net.train()
    
    class_Accuracy, mean_Accuracy = compute_Accuracy(cm)
    
    return class_Accuracy, mean_Accuracy


 

def IoU(mask_true, mask_pred, n_classes=2):
        
        labels = np.arange(n_classes)
        cm = confusion_matrix(mask_true.flatten(), mask_pred.flatten(), labels=labels)
        
        return compute_Accuracy(cm)