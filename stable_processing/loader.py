import os
from PIL import Image
from torch.nn import functional as F

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import Tuple


import numpy as np

'''
    This is the image loader for sam images. 
    after loading the sam images, we will process it using encoder to extract features in patch
'''

class TransformImage:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int =1024, device = 'cpu') -> None:
        self.target_length = target_length
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(device)
        
        self.image_size = target_length

    def apply_image(self, image: np.ndarray, device = 'cuda') -> torch.Tensor:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        input_image=  np.array(resize(to_pil_image(image), target_size))
        
        input_image_torch = torch.as_tensor(input_image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        input_image = self.preprocess(input_image_torch)
        return input_image
        
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
        
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    



    
class ImageDataset(Dataset):
    def __init__(self, directory, transform: TransformImage =None):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): A TransformImage instance or similar for processing images.
            device (string): Device to perform computations on.
        """
        self.directory = directory
        self.transform = transform  # Expecting an instance of TransformImage
        self.images = [os.path.join(directory, img) for img in sorted(os.listdir(directory)) if img.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(self.images)} images.")
        self.mask = self.generate_padding_mask()
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            image = self.transform.apply_image(image, 'cpu')
        
        basename = os.path.basename(img_path)
        return image, basename
    
    def generate_padding_mask(self) -> torch.Tensor:
        image = Image.open(self.images[0]).convert('RGB')
        image = np.array(image)
        h, w = image.shape[0], image.shape[1]
        """Generate a mask indicating the padding regions."""
        padh = self.transform.image_size - h
        padw = self.transform.image_size - w
        mask = torch.zeros((h, w))
        mask = F.pad(mask, (0, padw, 0, padh), value=1)
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        # Downscale the mask
        downscaled_mask = F.interpolate(mask, size=(64, 64), mode='nearest')
        # Remove the added dimensions and return as binary
        downscaled_mask = downscaled_mask.squeeze()
        return downscaled_mask


def load_dataset(directory, batch_size, num_workers):
    # Initialize the image transformation class
    transform = TransformImage()
    
    # Create dataset
    dataset = ImageDataset(directory, transform=transform)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return dataloader, len(dataset), dataset.mask.cuda()
# import os
# from PIL import Image, ImageFile
# from torch.nn import functional as F
# from torch.utils.data import Dataset, DataLoader
# import torch
# from torchvision.transforms.functional import resize, to_pil_image
# from typing import Tuple, List
# import numpy as np

# # 为防止 PIL 报错，设置 ImageFile.LOAD_TRUNCATED_IMAGES 为 True
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# class TransformImage:
#     """
#     Resizes images to longest side 'target_length', as well as provides
#     methods for resizing coordinates and boxes. Provides methods for
#     transforming both numpy array and batched torch tensors.
#     """

#     def __init__(self, target_length: int = 1024, device='cpu') -> None:
#         self.target_length = target_length
#         pixel_mean = [123.675, 116.28, 103.53]
#         pixel_std = [58.395, 57.12, 57.375]
        
#         self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).to(device)
#         self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).to(device)
        
#         self.image_size = target_length

#     def apply_image(self, image: np.ndarray, device='cuda') -> torch.Tensor:
#         """
#         Expects a numpy array with shape HxWxC in uint8 format.
#         """
#         target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
#         input_image = np.array(resize(to_pil_image(image), target_size))
        
#         input_image_torch = torch.as_tensor(input_image, device=device)
#         input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
#         input_image = self.preprocess(input_image_torch)
#         return input_image
        
#     def preprocess(self, x: torch.Tensor) -> torch.Tensor:
#         """Normalize pixel values and pad to a square input."""
#         # Normalize colors
#         x = (x - self.pixel_mean) / self.pixel_std

#         # Pad
#         h, w = x.shape[-2:]
#         padh = self.image_size - h
#         padw = self.image_size - w
#         x = F.pad(x, (0, padw, 0, padh))
#         return x
        
#     @staticmethod
#     def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
#         """
#         Compute the output size given input size and target long side length.
#         """
#         scale = long_side_length * 1.0 / max(oldh, oldw)
#         newh, neww = oldh * scale, oldw * scale
#         neww = int(neww + 0.5)
#         newh = int(newh + 0.5)
#         return (newh, neww)

# class ImageDataset(Dataset):
#     def __init__(self, directory: str, transform: TransformImage = None):
#         """
#         Args:
#             directory (string): Directory with all the images.
#             transform (callable, optional): A TransformImage instance or similar for processing images.
#         """
#         self.directory = directory
#         self.transform = transform
#         self.images = self._find_images(directory)
#         if not self.images:
#             raise ValueError(f"目录 {self.directory} 中没有找到图片。")
#         print(f"在目录 {self.directory} 中找到 {len(self.images)} 张图片：")
#         for img_path in self.images:
#             print(img_path)
#         self.mask = self.generate_padding_mask()
        
#     def _find_images(self, directory: str) -> List[str]:
#         """
#         Recursively find all images in the directory and its subdirectories.
#         """
#         image_extensions = ('.png', '.jpg', '.jpeg')
#         images = []
#         for root, _, files in os.walk(directory):
#             for file in files:
#                 if file.endswith(image_extensions):
#                     images.append(os.path.join(root, file))
#         return sorted(images)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path = self.images[idx]
#         try:
#             image = Image.open(img_path).convert('RGB')
#             image = np.array(image)
#         except (OSError, Image.DecompressionBombError) as e:
#             print(f"警告: 跳过损坏的图像 {img_path}. 错误: {e}")
#             # 跳过有问题的图片，返回一个占位图像或空图像
#             return self.__getitem__((idx + 1) % len(self.images))
        
#         if self.transform:
#             image = self.transform.apply_image(image, 'cpu')
        
#         basename = os.path.basename(img_path)
#         return image, basename
    
#     def generate_padding_mask(self) -> torch.Tensor:
#         image = Image.open(self.images[0]).convert('RGB')
#         image = np.array(image)
#         h, w = image.shape[0], image.shape[1]
#         """Generate a mask indicating the padding regions."""
#         padh = self.transform.image_size - h
#         padw = self.transform.image_size - w
#         mask = torch.zeros((h, w))
#         mask = F.pad(mask, (0, padw, 0, padh), value=1)
#         mask = mask.unsqueeze(0).unsqueeze(0).float()
#         # Downscale the mask
#         downscaled_mask = F.interpolate(mask, size=(64, 64), mode='nearest')
#         # Remove the added dimensions and return as binary
#         downscaled_mask = downscaled_mask.squeeze()
#         return downscaled_mask

# def load_dataset(directory: str, batch_size: int, num_workers: int):
#     # Initialize the image transformation class
#     transform = TransformImage()
    
#     # Create dataset
#     dataset = ImageDataset(directory, transform=transform)
    
#     # Create data loader
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
#     return dataloader, len(dataset), dataset.mask.cuda()
