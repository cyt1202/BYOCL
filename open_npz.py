import numpy as np
import cv2
import matplotlib.pyplot as plt
data = np.load('/home/xiongbutian/workspace/sc_latent_sam/output_files/Davis-script-adjust-output/bear.npz')
image = data['00000.jpg'][3] #* 255

# 可视化并保存二值掩码
plt.imshow(image, cmap='gray')  # 使用灰度图显示
plt.title("Binary Mask Visualization")

# 保存图像到文件
plt.savefig('/home/xiongbutian/workspace/sc_latent_sam/binary_mask_visualization.png')



# data = np.load('/home/xiongbutian/workspace/sc_latent_sam/output_files/Davis-script-adjust-output/bear/adjusted_results.npz')
# print("Keys in the npz file:", data.keys())

# # Access and print the array data
# for key in data.keys():

#     print(f"Data under key '{key}':\n", data[key])
#     print("Shape:", data[key].shape)
# data.close()

# # 检查 refined_label.npz 文件中的键
# refined_label_path = "/home/xiongbutian/workspace/sc_latent_sam/output_files/Davis-output/shooting/saved_batch_labels.npz"
# refined_mask_path = "/home/xiongbutian/workspace/sc_latent_sam/output_files/Davis-output/shooting/refined_mask.npz"

# # 打印 refined_label.npz 文件中的键
# with np.load(refined_label_path) as data:
#     print("refined_label.npz 中的键:", data.files)

# # 打印 refined_mask.npz 文件中的键
# with np.load(refined_mask_path) as data:
#     print("refined_mask.npz 中的键:", data.files)


