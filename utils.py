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
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    print(f"Unique values in pred: {np.unique(pred)}")
    print(f"Unique values in gt: {np.unique(gt)}")
    print(f"Pred sum: {pred.sum()}, GT sum: {gt.sum()}")
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        print(f"Dice: {dice}, HD95: {hd95}")
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        print("Pred sum > 0 but GT sum = 0")
        return 1, 0
    else:
        print("Pred sum = 0 or both Pred and GT sum = 0")
        return 0, 0

def test_single_volume(images, labels, net, classes, patch_size=[224, 224], test_save_path=None, cases=None, z_spacing=1):
    images, labels = images.cpu().detach().numpy(), labels.cpu().detach().numpy()
    batch_size, _, h, w = images.shape
    
    predictions = np.zeros((batch_size, patch_size[0], patch_size[1]))
    metric_lists = []
    
    for b in range(batch_size):
        image, label = images[b], labels[b]
        
        print(f"\nProcessing image {b}")
        print(f"Original image shape: {image.shape}, Original label shape: {label.shape}")
        print(f"Original image min: {image.min()}, max: {image.max()}")
        print(f"Original label unique values: {np.unique(label)}")
        
        # Resize both image and label to patch_size
        image = zoom(image, (1, patch_size[0] / h, patch_size[1] / w), order=3)
        label = zoom(label, (patch_size[0] / h, patch_size[1] / w), order=0)  # Use order=0 for nearest neighbor interpolation
        
        print(f"Resized image shape: {image.shape}, Resized label shape: {label.shape}")
        print(f"Resized image min: {image.min()}, max: {image.max()}")
        print(f"Resized label unique values: {np.unique(label)}")
        
        input = torch.from_numpy(image).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            print(f"Network output shape: {outputs.shape}")
            print(f"Network output min: {outputs.min().item()}, max: {outputs.max().item()}")
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = prediction.cpu().detach().numpy()
        
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction unique values: {np.unique(prediction)}")
        
        predictions[b] = prediction
        
        # Calculate metrics for this case
        metric_list = []
        for i in range(0, classes):
            print(f"\nCalculating metrics for class {i}")
            metric_list.append(calculate_metric_percase(prediction == i, label == i))
        metric_lists.append(metric_list)
        
        # Save debug visualizations
        if test_save_path is not None and cases is not None:
            case = cases[b]
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(image[0], cmap='gray')
            plt.title('Input Image')
            plt.subplot(132)
            plt.imshow(prediction, cmap='jet')
            plt.title('Prediction')
            plt.subplot(133)
            plt.imshow(label, cmap='jet')
            plt.title('Ground Truth')
            plt.savefig(f"{test_save_path}/{case}_debug.png")
            plt.close()
        
        # Save results if path is provided
        if test_save_path is not None and cases is not None:
            case = cases[b]
            img_itk = sitk.GetImageFromArray(image.astype(np.float32))
            prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
            lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
            img_itk.SetSpacing((1, 1, z_spacing))
            prd_itk.SetSpacing((1, 1, z_spacing))
            lab_itk.SetSpacing((1, 1, z_spacing))
            sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    
    print(f"\nFinal metric_lists: {metric_lists}")
    return metric_lists
