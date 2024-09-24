import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_keratinpearls import KeratinPearls_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/KeratinPearls/test_npz', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='KeratinPearls', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_KeratinPearls', help='list dir')

parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--model_path', type=str, default=None, help='path to a specific model to test')
args = parser.parse_args()

    
def inference(args, model, test_save_path=None):
    db_test = KeratinPearls_dataset(base_dir=args.root_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False, num_workers=8)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = []

    print(f"Dataset size: {len(db_test)}")
    print(f"Batch size: {args.batch_size}")

    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(testloader)):
            images, labels, case_names = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name']
            
            print(f"\nBatch {i_batch}:")
            print(f"  Original Images shape: {images.shape}")
            print(f"  Original Labels shape: {labels.shape}")
            print(f"  Images min: {images.min()}, max: {images.max()}")
            print(f"  Labels unique values: {torch.unique(labels)}")
            
            images = images.permute(0, 3, 1, 2)
            
            print(f"  Permuted Images shape: {images.shape}")
            print(f"  Number of cases: {len(case_names)}")

            metric_i = test_single_volume(images, labels, model, classes=args.num_classes, 
                                          patch_size=[args.img_size, args.img_size],
                                          test_save_path=test_save_path, cases=case_names, z_spacing=1)
            
            print(f"  Metric_i: {metric_i}")
            
            metric_list.extend(metric_i)
            
            for i, case_name in enumerate(case_names):
                logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % 
                             (i_batch * args.batch_size + i, case_name, 
                              np.mean(metric_i[i], axis=0)[0], np.mean(metric_i[i], axis=0)[1]))

    metric_list = np.array(metric_list)
    print(f"Final metric_list shape: {metric_list.shape}")
    print(f"Final metric_list: {metric_list}")

    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % 
                     (i, np.mean(metric_list[:, i-1, 0]), np.mean(metric_list[:, i-1, 1])))

    performance = np.mean(metric_list, axis=(0, 1))[0]
    mean_hd95 = np.mean(metric_list, axis=(0, 1))[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

    return "Testing Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'KeratinPearls': {
            'Dataset': KeratinPearls_dataset,
            'root_path': '../data/KeratinPearls/',
            'list_dir': './lists/lists_KeratinPearls',
            'num_classes': 2,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    if args.model_path:
        snapshot = args.model_path

        if not os.path.exists(snapshot):
            raise FileNotFoundError(f"Model file not found: {snapshot}")

        net.load_state_dict(torch.load(snapshot))
        snapshot_name = os.path.basename(snapshot)
    else: 
        snapshot = os.path.join(snapshot_path, 'best_model.pth')

        if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
        net.load_state_dict(torch.load(snapshot))
        snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)

