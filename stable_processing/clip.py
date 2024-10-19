import torch
import torch.nn.functional as F
from torchvision import transforms
from clip import clip

# 假设你已经有了 ViT 的特征，形状为 (B, H, W, C)
vit_features = ...  # 例如 (B, H, W, C)

# 使用平均池化将特征转换为 (B, C) 的形状
# 这里使用全局平均池化
B, H, W, C = vit_features.shape
vit_features_pooled = F.adaptive_avg_pool2d(vit_features.permute(0, 3, 1, 2), (1, 1)).view(B, C)

# 加载 CLIP 模型
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

original_images = image

# 计算 CLIP 的图像特征
image_inputs = torch.stack([preprocess(img) for img in original_images]).to("cuda")
with torch.no_grad():
    clip_image_features = clip_model.encode_image(image_inputs)

text_descriptions = ["description of the image", "another description", ...]  # 根据实际情况填充
text_inputs = clip.tokenize(text_descriptions).to("cuda")

with torch.no_grad():
    clip_text_features = clip_model.encode_text(text_inputs)

# 将 ViT 的特征与 CLIP 的特征结合
combined_features = torch.cat((vit_features_pooled, clip_image_features), dim=1)  # 这里假设维度匹配

# softmax
combined_features = torch.nn.functional.normalize(combined_features, dim=1)

print(combined_features.shape)  # 输出融合后的特征形状
