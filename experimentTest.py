# from skimage import io
# from sklearn.metrics import jaccard_score, f1_score
# from skimage.io import imread
# import os
# import numpy as np

# # 定义路径
# # segmentation_base_path = "/home/xiongbutian/workspace/sc_latent_sam/output_files/5ebef6bd"
# # ground_truth_base_path = "/home/xiongbutian/workspace/Sam2/segment-anything-2/train/Annotations/5ebef6bd"
# segmentation_base_path = "/home/xiongbutian/workspace/sc_latent_sam/output_files/output8"
# ground_truth_base_path = "/home/xiongbutian/workspace/davis2017-evaluation/DAVIS/Annotations/480p/blackswan"
# # 初始化用于计算总体指标的列表
# iou_scores = []
# dice_scores = []
# pixel_accuracies = []
# j_scores = []
# f_scores = []
# # 遍历每个图像
# # for i in range(6767):
# for i in range(50):
#     # 定义子文件夹和文件路径
#     folder_name = f"{i:05d}"
#     ground_truth_image_path = os.path.join(ground_truth_base_path, f"{folder_name}.png")
    
#     segmentation_folder_path = os.path.join(segmentation_base_path, folder_name)
    
#     # 读取 ground truth 图像
#     ground_truth = imread(ground_truth_image_path, as_gray=True)
    
#     best_iou = 0
#     best_segmentation = None

#     # 遍历每个分割结果
#     for j in range(6):
#         segmentation_image_path = os.path.join(segmentation_folder_path, f"{j}.png")
#         segmentation_result = imread(segmentation_image_path, as_gray=True)

#         # 将图像数据转换为二进制形式 (0 和 1)
#         threshold = 0.5
#         ground_truth_binary = (ground_truth > threshold).astype(int)
#         segmentation_binary = (segmentation_result > threshold).astype(int)

#         # 计算 IoU (Intersection over Union)
#         intersection = np.logical_and(ground_truth_binary, segmentation_binary)
#         union = np.logical_or(ground_truth_binary, segmentation_binary)
#         iou = np.sum(intersection) / np.sum(union)

#         # 更新最佳 IoU 和最佳分割结果
#         if iou > best_iou:
#             best_iou = iou
#             best_segmentation = segmentation_binary

#     # 计算其他指标
#     dice = f1_score(ground_truth_binary.flatten(), best_segmentation.flatten())
#     pixel_accuracy = np.sum(ground_truth_binary == best_segmentation) / ground_truth_binary.size
#     j_score = jaccard_score(ground_truth_binary.flatten(), best_segmentation.flatten())
#     f_score = f1_score(ground_truth_binary.flatten(), best_segmentation.flatten())

#     # 存储指标
#     iou_scores.append(best_iou)
#     dice_scores.append(dice)
#     pixel_accuracies.append(pixel_accuracy)
#     j_scores.append(j_score)
#     f_scores.append(f_score)
# # 计算总体指标的平均值
# average_iou = np.mean(iou_scores)
# average_dice = np.mean(dice_scores)
# average_pixel_accuracy = np.mean(pixel_accuracies)
# average_j_score = np.mean(j_scores)
# average_f_score = np.mean(f_scores)

# # 输出结果
# print(f"平均 IoU: {average_iou}")
# print(f"平均 Dice 系数: {average_dice}")
# print(f"平均像素精度: {average_pixel_accuracy}")
# print(f"平均 J 分数: {average_j_score}")
# print(f"平均 F 分数: {average_f_score}")


from skimage import io
from sklearn.metrics import jaccard_score, f1_score
from skimage.io import imread
import os
import numpy as np

# 定义路径
segmentation_base_path = "/home/xiongbutian/workspace/sc_latent_sam/output_files/output8"
ground_truth_base_path = "/home/xiongbutian/workspace/davis2017-evaluation/DAVIS/Annotations/480p/blackswan"

# 初始化用于计算总体指标的列表
iou_scores = []
dice_scores = []
pixel_accuracies = []
j_scores = []
f_scores = []

# 遍历每个图像
for i in range(50):
    # 定义子文件夹和文件路径
    folder_name = f"{i:05d}"
    ground_truth_image_path = os.path.join(ground_truth_base_path, f"{folder_name}.png")
    segmentation_folder_path = os.path.join(segmentation_base_path, folder_name)

    # 读取 ground truth 图像
    try:
        ground_truth = imread(ground_truth_image_path, as_gray=True)
    except Exception as e:
        print(f"无法读取 ground truth 图像: {ground_truth_image_path}, 错误: {e}")
        continue

    best_iou = 0
    best_segmentation = None

    # 遍历每个分割结果
    for j in range(6):
        segmentation_image_path = os.path.join(segmentation_folder_path, f"{j}.png")
        if not os.path.exists(segmentation_image_path):
            continue  # 如果图像不存在，则跳过

        try:
            segmentation_result = imread(segmentation_image_path, as_gray=True)
        except Exception as e:
            print(f"无法读取分割图像: {segmentation_image_path}, 错误: {e}")
            continue

        # 将图像数据转换为二进制形式 (0 和 1)
        threshold = 0.5
        ground_truth_binary = (ground_truth > threshold).astype(int)
        segmentation_binary = (segmentation_result > threshold).astype(int)

        # 计算 IoU (Intersection over Union)
        intersection = np.logical_and(ground_truth_binary, segmentation_binary)
        union = np.logical_or(ground_truth_binary, segmentation_binary)
        
        # 避免除以零的情况
        if np.sum(union) == 0:
            iou = 0
        else:
            iou = np.sum(intersection) / np.sum(union)

        # 更新最佳 IoU 和最佳分割结果
        if iou > best_iou:
            best_iou = iou
            best_segmentation = segmentation_binary

    # 确保在计算指标之前 best_segmentation 不是 None
    if best_segmentation is not None:
        dice = f1_score(ground_truth_binary.flatten(), best_segmentation.flatten())
        pixel_accuracy = np.sum(ground_truth_binary == best_segmentation) / ground_truth_binary.size
        j_score = jaccard_score(ground_truth_binary.flatten(), best_segmentation.flatten())
        f_score = f1_score(ground_truth_binary.flatten(), best_segmentation.flatten())

        # 存储指标
        iou_scores.append(best_iou)
        dice_scores.append(dice)
        pixel_accuracies.append(pixel_accuracy)
        j_scores.append(j_score)
        f_scores.append(f_score)
    else:
        print(f"对于图像 {folder_name}，未找到有效的分割结果")

# 计算总体指标的平均值
average_iou = np.mean(iou_scores) if iou_scores else 0
average_dice = np.mean(dice_scores) if dice_scores else 0
average_pixel_accuracy = np.mean(pixel_accuracies) if pixel_accuracies else 0
average_j_score = np.mean(j_scores) if j_scores else 0
average_f_score = np.mean(f_scores) if f_scores else 0

# 输出结果
print(f"平均 IoU: {average_iou}")
print(f"平均 Dice 系数: {average_dice}")
print(f"平均像素精度: {average_pixel_accuracy}")
print(f"平均 J 分数: {average_j_score}")
print(f"平均 F 分数: {average_f_score}")

# import os
# import numpy as np
# from skimage.io import imread
# from sklearn.metrics import f1_score, jaccard_score
# import warnings
# from tqdm import tqdm

# def calculate_metrics(gt, pred):
#     # 计算二值图的交集和并集
#     intersection = np.logical_and(gt, pred)
#     union = np.logical_or(gt, pred)
#     # 计算IoU
#     print(np.sum(union))
#     if np.sum(union) != 0:
#         iou = np.sum(intersection) / np.sum(union)
#     else:
#         0
    
#     # 计算Dice系数
#     dice = 2 * np.sum(intersection) / (np.sum(gt) + np.sum(pred)) if (np.sum(gt) + np.sum(pred)) != 0 else 0

#     # 计算像素精度
#     pixel_accuracy = np.sum(gt == pred) / gt.size

#     # 计算Jaccard指数和F1分数
#     j_score = jaccard_score(gt.flatten(), pred.flatten())
#     f_score = f1_score(gt.flatten(), pred.flatten())

#     return iou, dice, pixel_accuracy, j_score, f_score

# def main(segmentation_base_path, ground_truth_base_path, num_images=1):
#     # 初始化用于计算总体指标的列表
#     metrics = {
#         'iou': [], 'dice': [], 'pixel_accuracy': [], 'j_score': [], 'f_score': []
#     }
#     image_range = tqdm(range(num_images), desc="Processing images")
#     # 遍历每个图像
#     for i in image_range:
#         folder_name = f"{i:05d}"
#         ground_truth_image_path = os.path.join(ground_truth_base_path, f"{folder_name}.png")
#         segmentation_folder_path = os.path.join(segmentation_base_path, folder_name)

#         if not os.path.exists(ground_truth_image_path):
#             warnings.warn(f"Ground truth image not found: {ground_truth_image_path}")
#             continue

#         # 读取ground truth图像
#         ground_truth = imread(ground_truth_image_path, as_gray=True) > 0.5  # 直接转换为二值图像

#         best_iou = 0
#         best_metrics = None

#         # 遍历每个分割结果
#         for j in range(5):
#             segmentation_image_path = os.path.join(segmentation_folder_path, f"{j}.png")
#             if not os.path.exists(segmentation_image_path):
#                 continue

#             segmentation_result = imread(segmentation_image_path, as_gray=True) > 0.5  # 直接转换为二值图像

#             # 计算指标
#             iou, dice, pixel_accuracy, j_score, f_score = calculate_metrics(ground_truth, segmentation_result)

#             # 更新最佳IoU和相关指标
#             if iou > best_iou:
#                 best_iou = iou
#                 best_metrics = (dice, pixel_accuracy, j_score, f_score)

#         if best_metrics:
#             # 存储指标
#             metrics['iou'].append(best_iou)
#             metrics['dice'].append(best_metrics[0])
#             metrics['pixel_accuracy'].append(best_metrics[1])
#             metrics['j_score'].append(best_metrics[2])
#             metrics['f_score'].append(best_metrics[3])

#     # 计算总体指标的平均值
#     for key in metrics:
#         print(f"平均 {key}: {np.mean(metrics[key]) if metrics[key] else 0}")

# if __name__ == '__main__':
#     segmentation_base_path = "/home/xiongbutian/workspace/sc_latent_sam/output_files/5ebef6bd"
#     ground_truth_base_path = "/home/xiongbutian/workspace/Sam2/segment-anything-2/train/Annotations/5ebef6bd"
#     main(segmentation_base_path, ground_truth_base_path)
