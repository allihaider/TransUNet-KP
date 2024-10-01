import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(int)
    gt = (gt > 0).astype(int)
    
    max_hd = np.sqrt(np.sum(np.array(pred.shape)**2))
    
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        print(f"Dice: {dice}, HD95: {hd95}")
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        print("Pred sum > 0 but GT sum = 0")
        return 0, max_hd 
    elif pred.sum() == 0 and gt.sum() > 0:
        print("GT sum > 0 but Pred sum = 0")
        return 0, max_hd  
    else:
        print("Both Pred and GT sum = 0")
        return 1, 0 


def test_single_volume(images, labels, net, classes, patch_size=[224, 224], test_save_path=None, cases=None, z_spacing=1):
    images, labels = images.cuda(), labels.cuda()
    net.eval()
    
    with torch.no_grad():
        outputs = net(images)
        predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
        
    metric_lists = []

    for b in range(images.shape[0]):
        metric_list = []

        for i in range(classes):
            metric_list.append(calculate_metric_percase(predictions[b] == i, labels[b] == i))
        metric_lists.append(metric_list)

        # Save debug visualizations
        if test_save_path is not None and cases is not None:
            case = cases[b]
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(images[b, 0], cmap='gray')
            plt.title('Input Image')
            plt.subplot(132)
            plt.imshow(predictions[b], cmap='jet')
            plt.title('Prediction')
            plt.subplot(133)
            plt.imshow(labels[b], cmap='jet')
            plt.title('Ground Truth')
            plt.savefig(f"{test_save_path}/{case}_debug.png")
            plt.close()
        
            img_itk = sitk.GetImageFromArray(images[b].astype(np.float32))
            prd_itk = sitk.GetImageFromArray(predictions[b].astype(np.float32))
            lab_itk = sitk.GetImageFromArray(labels[b].astype(np.float32))
            img_itk.SetSpacing((1, 1, z_spacing))
            prd_itk.SetSpacing((1, 1, z_spacing))
            lab_itk.SetSpacing((1, 1, z_spacing))
            sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    
    return metric_lists

