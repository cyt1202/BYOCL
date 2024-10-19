import numpy as np
import torch
import torch.nn.functional as F

# 加载 npz 文件
data = np.load('/home/xiongbutian/workspace/sc_latent_sam/output_files/Davis-script-output/bear/refined_mask.npz')
print("Keys in the npz file:", data.keys())

# 加载 ground truth (gt) 文件
gt = np.load('/home/xiongbutian/workspace/sc_latent_sam/Annotations/Davis/bear.npz')

# 定义 logits_to_mask 函数
def logits_to_mask(logits: torch.Tensor, threshold=0.5) -> torch.Tensor:
    '''
    Basic process strategy that can smooth the segmentation.
    '''
    # 创建简单的均值卷积核
    kernel_size = 4  # 卷积核大小 (4x4)
    kernel = torch.ones((kernel_size, kernel_size)).cuda() / (kernel_size ** 2)
    kernel = kernel.expand((1, 1, kernel_size, kernel_size))  # 扩展以适配 conv2d

    # 确保 kernel 是浮点张量
    kernel = kernel.type(torch.float32)
    
    # 添加 batch 维度并重新排列
    logits = logits.unsqueeze(0)
    logits = logits.permute(1, 0, 2, 3)
    
    # 使用卷积核过滤 logits
    filtered_logits = F.conv2d(logits, kernel, padding=1, stride=1)
    
    # 使用阈值生成二值化张量
    binary_tensor = torch.where(filtered_logits < threshold, torch.tensor(0.0).cuda(), torch.tensor(1.0).cuda())
    
    return binary_tensor

# 定义 alpha_mask_generation 函数
def alpha_mask_generation(original_image_shape: tuple, masks: torch.Tensor) -> torch.Tensor:
    '''
    Resize the mask to match the original image size after processing.
    '''
    # 将 masks 转换为二值化张量
    masks = logits_to_mask(masks)
    
    # 计算最大尺寸
    max_size = max(original_image_shape)

    # 使用 nearest neighbor 插值进行调整
    image_large = F.interpolate(masks, size=(max_size, max_size), mode='nearest')
    image_large = image_large.squeeze()

    # 调整维度
    if image_large.dim() == 2:
        image_large = image_large.unsqueeze(0)  # 添加维度，确保为 3D 张量

    image_large = image_large.permute(0, 2, 1)  # 调整维度顺序

    # 获取原始图片的高度和宽度
    original_height, original_width = original_image_shape

    # 移除填充以匹配原始尺寸
    if original_height == max(original_image_shape):
        return image_large[:, :, :original_width]
    else:
        return image_large[:, :original_height, :]

# 初始化存储 binary tensor 的字典
binary_tensors = {}

for key in gt.keys():
    original_image_shape = gt[key].shape

# 遍历 npz 文件的内容并处理每个数组
for key, value in data.items():
    print(f"Processing key: {key}")
    
    # 将 npz 文件中的值转换为 torch.Tensor
    logits = torch.tensor(value).cuda()  # 假设 value 是 numpy 数组
    
    # 调用 alpha_mask_generation 函数生成调整后的 masks
    binary_tensor = alpha_mask_generation(original_image_shape, logits)
    
    # 将结果存储在字典中
    binary_tensors[key] = binary_tensor.cpu().numpy()  # 将结果转换回 numpy 并保存

# 现在 binary_tensors 字典中存储了处理后的所有二值化结果
for key, tensor in binary_tensors.items():
    print(f"Processed binary tensor for key '{key}' with shape: {tensor.shape}")
