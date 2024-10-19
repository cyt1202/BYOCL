import numpy as np
import torch
import torch.nn.functional as F
import argparse

# 定义 logits_to_mask 函数
def logits_to_mask(logits: torch.Tensor, threshold=0.5) -> torch.Tensor:
    '''
    Basic process strategy that can smooth the segmentation.
    '''
    kernel_size = 4  # 卷积核大小 (4x4)
    kernel = torch.ones((kernel_size, kernel_size)).cuda() / (kernel_size ** 2)
    kernel = kernel.expand((1, 1, kernel_size, kernel_size))  # 扩展以适配 conv2d
    kernel = kernel.type(torch.float32)
    
    logits = logits.unsqueeze(0)
    logits = logits.permute(1, 0, 2, 3)
    
    filtered_logits = F.conv2d(logits, kernel, padding=1, stride=1)
    
    binary_tensor = torch.where(filtered_logits < threshold, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
    
    return binary_tensor

# 定义 alpha_mask_generation 函数
def alpha_mask_generation(original_image_shape: tuple, masks: torch.Tensor) -> torch.Tensor:
    '''
    Resize the mask to match the original image size after processing.
    '''
    masks = logits_to_mask(masks)
    max_size = max(original_image_shape)
    image_large = F.interpolate(masks, size=(max_size, max_size), mode='nearest')
    image_large = image_large.squeeze()

    if image_large.dim() == 2:
        image_large = image_large.unsqueeze(0)
    
    original_height, original_width = original_image_shape
    if original_height == max(original_image_shape):
        return image_large[:, :, :original_width]
    else:
        return image_large[:, :original_height, :]

def main(data_path, gt_path, output_dir):
    # 加载 npz 文件
    data = np.load(data_path)
    print("Keys in the npz file:", data.keys())

    # 加载 ground truth (gt) 文件
    gt = np.load(gt_path)

    # 初始化存储 binary tensor 的字典
    binary_tensors = {}

    for key in gt.keys():
        original_image_shape = gt[key].shape

        # 遍历 npz 文件的内容并处理每个数组
        for k, value in data.items():
            logits = torch.tensor(value).cuda()
            binary_tensor = alpha_mask_generation(original_image_shape, logits)
            binary_tensors[k] = binary_tensor.cpu().numpy()

    # 保存处理后的二值化结果为 npz 文件
    output_path = f"{output_dir}.npz"
    np.savez_compressed(output_path, **binary_tensors)
    print(f"Processed results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some npz files.")
    parser.add_argument('--data', type=str, required=True, help='Path to the npz file containing logits.')
    parser.add_argument('--gt', type=str, required=True, help='Path to the npz file containing ground truth masks.')
    parser.add_argument('--output', type=str, required=True, help='Output directory to save processed results.')
    
    args = parser.parse_args()
    main(args.data, args.gt, args.output)
