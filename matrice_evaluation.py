import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
from skimage import io
import torch
import torch.nn.functional as F

import argparse



def parser():
    parser = argparse.ArgumentParser("Get gt npz file", add_help=True)
    parser.add_argument("--gt_location", "-g", type=str, required=True, help="Our mask direction")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="Output directory")
    # parser.add_argument("--gt_npz", "-g_npz", type=str, required=True, help="GT npz direction")
    # parser.add_argument("--pred_npz", "-p_npz", type=str, required=True, help="pred npz direction")
    
    args = parser.parse_args()
    return args


def copyGT(gt,pred):
    copiedGT = np.array() #创建数组
    B = pred.shape[0] #复制GT的次数
    for i in range(B):
        copiedGT[i] = gt
    return copiedGT #返回（B，H，W）的GT数组
    
def calIoU(gt_dir, pred_dir):
    #打开gt,pred的npz文件
    gts = np.load(gt_dir)
    preds = np.load(pred_dir)
    counter = 0
    IoU_list = []
    for key, value in gts.items():
        new_key = key.split('.')[0]
        pred = list(preds)[counter].value
        gts[new_key] = copyGT(value,pred)
        # AND = np.and(gts[new_key],pred)

def main():
    args = parser()
    
    gt_location = args.gt_location
    output_dir = args.output_dir
    # gt_dir = args.gt_npz
    # pred_dir = args.pred_npz
    
    # calIoU(gt_dir, pred_dir)
    gt_masks_dict={}
    
    for gt_mask_file in sorted(os.listdir(gt_location)):
        gt_mask_path = os.path.join(gt_location, gt_mask_file)
        seq_name = os.path.splitext(gt_mask_file)[0]
        gt_mask = io.imread(gt_mask_path)
        grayscale_gt_mask = 0.2989 * gt_mask[:, :, 0] + 0.5870 * gt_mask[:, :, 1] + 0.1140 * gt_mask[:, :, 2]
        binary_gt_mask = (grayscale_gt_mask> 0.5).astype(int)
        gt_masks_dict.update({f'{seq_name}':binary_gt_mask})

    np.savez_compressed(output_dir, **gt_masks_dict)   
    
        
if __name__ == '__main__':
    main()