from stable_processing.decode_mask import black_overlay, alpha_mask_generation, hard_mask,color_mask
from stable_processing.logging import print_with_color
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
import os
import torch

import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_image_height_width(image_location):
    img = Image.open(image_location)
    img = np.asarray(img)
    return img.shape[:2]

def get_source_file(npz_location:str, img_dir:str):
    masks = np.load(npz_location)
    img_list = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    return masks, img_list

def generate_image_group(masks:torch.Tensor, original_image:str, stored_location:str):
    original_image = Image.open((original_image)).convert('RGB')
    count = 0
    alpha_mask = alpha_mask_generation(original_image.size, masks)

    alpha_mask = alpha_mask.permute(0,2,1)

    alpha_mask:np.ndarray = alpha_mask.cpu().numpy()
    
    blended_images = hard_mask(alpha_mask,original_image)
    # print('blended_images shpae:', len(blended_images))
    os.makedirs(stored_location,exist_ok=True)
    
    
    # exit()
    
    for image in blended_images:
        image.save(f'{stored_location}/{count}.png')
        count += 1
        image.close()
    
    

    # for image in blended_images:
    #     image.save(f'{stored_location}/{count}.png')
    #     count += 1
    #     image.close()
    
    # for image in blended_images:
    #     image.save(f'{stored_location}.png')
    #     image.close()
    

    

def parser():
    parser = argparse.ArgumentParser("Visualize The Maksed Result", add_help=True)
    parser.add_argument("--image_dir", "-i", type=str, required=True, help="path to image folder")
    parser.add_argument("--mask_location", "-m", type=str, required=True, help="Our mask direction")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="Output directory")
    args = parser.parse_args()
    return args

def main():
    args = parser()
    
    mask_location = args.mask_location
    image_dir = args.image_dir
    output_dir = args.output_dir
    print_with_color(f'the output directory is {output_dir}', 'YELLOW')
    print_with_color(f'the mask_location is {mask_location}', 'YELLOW')
    print_with_color(f'the image_dir is {image_dir}', 'YELLOW')
    
    if os.path.exists(output_dir):
        print_with_color(f'The output dir exists, skip creating directory', 'YELLOW')
    else: 
        os.makedirs(output_dir)
        print_with_color(f'The output dir does not exist, creating directory', 'YELLOW')

    masks, img_list = get_source_file(mask_location, image_dir)


    print_with_color('overlaing image ...', 'YELLOW')
    for i, image_location in tqdm(enumerate(img_list), total=len(img_list), desc="Processing items"):
        key = image_location.split('/')[-1]
        image_location = key.split('.')[0]
        
        generate_image_group(
                            torch.Tensor(masks[key]).cuda(), 
                            os.path.join(image_dir, key), 
                            os.path.join(output_dir,image_location)
                            )
        print(f'mask{key}')
        
    print_with_color(f'Visualization is accomplished, the data is stored in {output_dir}', 'GREEN')

if __name__ == '__main__':
    main()





# from stable_processing.decode_mask import black_overlay, alpha_mask_generation, hard_mask
# from stable_processing.logging import print_with_color
# from tqdm import tqdm
# from PIL import Image
# import numpy as np
# import os
# import torch

# import argparse

# def get_image_height_width(image_location):
#     img = Image.open(image_location)
#     img = np.asarray(img)
#     return img.shape[:2]

# def get_source_file(npz_location:str, img_dir:str):
#     masks = np.load(npz_location)
#     img_list = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
#     return masks, img_list

# def generate_image_group(masks:torch.Tensor, original_image:str, stored_location:str):
#     original_image = Image.open((original_image)).convert('RGB')
#     count = 0
#     alpha_mask = alpha_mask_generation(original_image.size, masks)

#     alpha_mask = alpha_mask.permute(0,2,1)

#     alpha_mask = alpha_mask.cpu().numpy()
#     #blended_images = black_overlay(alpha_mask, original_image) # blended images B, H, W
#     blended_images = hard_mask(alpha_mask, original_image)
#     os.makedirs(stored_location,exist_ok=True)
#     for image in blended_images:
#         image.save(f'{stored_location}/{count}.png')
#         count += 1
#         image.close()
    

# def parser():
#     parser = argparse.ArgumentParser("Visualize The Maksed Result", add_help=True)
#     parser.add_argument("--image_dir", "-i", type=str, required=True, help="path to image folder")
#     parser.add_argument("--mask_location", "-m", type=str, required=True, help="Our mask direction")
#     parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="Output directory")
#     args = parser.parse_args()
#     return args

# def main():
#     args = parser()
    
#     mask_location = args.mask_location
#     image_dir = args.image_dir
#     output_dir = args.output_dir
#     print_with_color(f'the output directory is {output_dir}', 'YELLOW')
#     print_with_color(f'the mask_location is {mask_location}', 'YELLOW')
#     print_with_color(f'the image_dir is {image_dir}', 'YELLOW')
    
#     if os.path.exists(output_dir):
#         print_with_color(f'The output dir exists, skip creating directory', 'YELLOW')
#     else: 
#         os.makedirs(output_dir)
#         print_with_color(f'The output dir does not exist, creating directory', 'YELLOW')

#     masks, img_list = get_source_file(mask_location, image_dir)


#     print_with_color('overlaing image ...', 'YELLOW')
#     for i, image_location in tqdm(enumerate(img_list), total=len(img_list), desc="Processing items"):
#         key = image_location.split('/')[-1]
#         image_location = key.split('.')[0]
        
#         generate_image_group(
#                             torch.Tensor(masks[key]).cuda(), 
#                             os.path.join(image_dir, key), 
#                             os.path.join(output_dir,image_location)
#                             )
#     print_with_color(f'Visualization is accomplished, the data is stored in {output_dir}', 'GREEN')

# if __name__ == '__main__':
#     main()
    
    
# from stable_processing.decode_mask import black_overlay, alpha_mask_generation, hard_mask
# from stable_processing.logging import print_with_color
# from tqdm import tqdm
# from PIL import Image
# import numpy as np
# import os
# import torch
# import argparse

# def get_image_height_width(image_location):
#     img = Image.open(image_location)
#     img = np.asarray(img)
#     return img.shape[:2]

# def get_source_file(npz_location:str, img_dir:str):
#     masks = {}
#     img_list = []

#     # 检查是否为目录
#     if os.path.isdir(img_dir):
#         # 获取所有子目录
#         subdirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
#         for subdir in subdirs:
#             subdir_path = os.path.join(img_dir, subdir)
#             # 获取子目录中的所有图像文件
#             imgs = [os.path.join(subdir_path, img) for img in sorted(os.listdir(subdir_path)) if img.endswith(('.png', '.jpg', '.jpeg'))]
#             img_list.extend(imgs)

#     # 加载 npz 文件
#     try:
#         masks_data = np.load(npz_location, allow_pickle=True)
#         for key in masks_data.files:
#             masks[key] = masks_data[key]
#     except Exception as e:
#         print(f"Error loading .npz file: {e}")
    
#     return masks, img_list

# def generate_image_group(masks:torch.Tensor, original_image:str, stored_location:str):
#     original_image = Image.open(original_image).convert('RGB')
#     count = 0
#     alpha_mask = alpha_mask_generation(original_image.size, masks)

#     alpha_mask = alpha_mask.permute(0,2,1)

#     alpha_mask = alpha_mask.cpu().numpy()
#     # blended_images = black_overlay(alpha_mask, original_image) # blended images B, H, W
#     blended_images = hard_mask(alpha_mask, original_image)
#     os.makedirs(stored_location, exist_ok=True)
#     for image in blended_images:
#         image.save(f'{stored_location}/{count}.png')
#         count += 1
#         image.close()

# def parser():
#     parser = argparse.ArgumentParser("Visualize The Masked Result", add_help=True)
#     parser.add_argument("--image_dir", "-i", type=str, required=True, help="Path to image folder")
#     parser.add_argument("--mask_location", "-m", type=str, required=True, help="Our mask location")
#     parser.add_argument("--output_dir", "-o", type=str, default="outputs", help="Output directory")
#     args = parser.parse_args()
#     return args

# def main():
#     args = parser()
    
#     mask_location = args.mask_location
#     image_dir = args.image_dir
#     output_dir = args.output_dir
#     print_with_color(f'the output directory is {output_dir}', 'YELLOW')
#     print_with_color(f'the mask_location is {mask_location}', 'YELLOW')
#     print_with_color(f'the image_dir is {image_dir}', 'YELLOW')
    
#     if os.path.exists(output_dir):
#         print_with_color(f'The output dir exists, skip creating directory', 'YELLOW')
#     else: 
#         os.makedirs(output_dir)
#         print_with_color(f'The output dir does not exist, creating directory', 'YELLOW')

#     masks, img_list = get_source_file(mask_location, image_dir)

#     print_with_color('Overlapping images...', 'YELLOW')
#     for i, image_location in tqdm(enumerate(img_list), total=len(img_list), desc="Processing items"):
#         subdir_name = os.path.basename(os.path.dirname(image_location))
#         output_subdir = os.path.join(output_dir, subdir_name)
#         os.makedirs(output_subdir, exist_ok=True)
#         key = os.path.basename(image_location).split('.')[0]
        
#         if key in masks:
#             generate_image_group(
#                                 torch.Tensor(masks[key]).cuda(), 
#                                 image_location, 
#                                 output_subdir
#                                 )
#         else:
#             print_with_color(f'Warning: Key {key} not found in masks dictionary.', 'RED')
    
#     print_with_color(f'Visualization is accomplished, the data is stored in {output_dir}', 'GREEN')

# if __name__ == '__main__':
#     main()

# from stable_processing.decode_mask import black_overlay, alpha_mask_generation, hard_mask
# from stable_processing.logging import print_with_color
# from tqdm import tqdm
# from PIL import Image
# import numpy as np
# import os
# import torch

# import argparse

# def get_image_height_width(image_location):
#     img = Image.open(image_location)
#     img = np.asarray(img)
#     return img.shape[:2]

# def get_source_files(img_dir:str):
#     subfolders = [os.path.join(img_dir, d) for d in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, d))]
#     mask_files = []
#     img_lists = []

#     for subfolder in subfolders:
#         masks_file = os.path.join(subfolder, 'refined_mask.npz')
#         if os.path.exists(masks_file):
#             masks = np.load(masks_file)
#             img_list = [os.path.join(subfolder, img) for img in sorted(os.listdir(subfolder)) if img.endswith(('.png', '.jpg', '.jpeg'))]
#             mask_files.append(masks)
#             img_lists.append(img_list)
    
#     return mask_files, img_lists, subfolders

# def generate_image_group(masks:torch.Tensor, original_image:str, stored_location:str):
#     original_image = Image.open((original_image)).convert('RGB')
#     count = 0
#     alpha_mask = alpha_mask_generation(original_image.size, masks)

#     alpha_mask = alpha_mask.permute(0,2,1)

#     alpha_mask = alpha_mask.cpu().numpy()
#     blended_images = hard_mask(alpha_mask, original_image)
#     os.makedirs(stored_location, exist_ok=True)
#     for image in blended_images:
#         image.save(f'{stored_location}/{count}.png')
#         count += 1
#         image.close()
    

# def parser():
#     parser = argparse.ArgumentParser("Visualize The Masked Result", add_help=True)
#     parser.add_argument("--image_dir", "-i", type=str, required=True, help="path to image folder")
#     parser.add_argument("--mask_location", "-m", type=str, required=True, help="Path to mask file (.npz)")
#     parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="Output directory")
#     args = parser.parse_args()
#     return args

# def main():
#     args = parser()
    
#     image_dir = args.image_dir
#     output_dir = args.output_dir

#     print_with_color(f'The output directory is {output_dir}', 'YELLOW')
#     print_with_color(f'The image directory is {image_dir}', 'YELLOW')
    
#     if os.path.exists(output_dir):
#         print_with_color(f'The output dir exists, skip creating directory', 'YELLOW')
#     else: 
#         os.makedirs(output_dir)
#         print_with_color(f'The output dir does not exist, creating directory', 'YELLOW')

#     mask_files, img_lists, subfolders = get_source_files(image_dir)

#     print_with_color('Overlaying images ...', 'YELLOW')
#     for masks, img_list, subfolder in zip(mask_files, img_lists, subfolders):
#         subfolder_name = os.path.basename(subfolder)
#         subfolder_output_dir = os.path.join(output_dir, subfolder_name)
#         for i, image_location in tqdm(enumerate(img_list), total=len(img_list), desc=f"Processing {subfolder_name}"):
#             key = os.path.basename(image_location)
#             image_name = key.split('.')[0]
#             generate_image_group(
#                 torch.Tensor(masks[key]).cuda(), 
#                 image_location, 
#                 os.path.join(subfolder_output_dir, image_name)
#             )

#     print_with_color(f'Visualization is accomplished, the data is stored in {output_dir}', 'GREEN')

# if __name__ == '__main__':
#     main()
