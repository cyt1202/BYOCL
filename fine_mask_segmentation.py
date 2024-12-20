'''
    We use the clustering mask as the input to the decoder
    Therefore the latent we have used in batch should be preserve

    Here is the overall stratgy
    - We follow the clustering feature partsm we get the lattent, and we preserve it in some where
    - We get the masks, and store it in some where, 
    - We use the masks as the input and send it to the decoder model to gether with the features
'''

'''
    We find that SAM cannot use dense mask as input along, we need to first convert it to bounding box
    And then we fake the mask's logits as the input and convert masks to bounding box
'''

from segment_anything.modeling.sam import Sam
from segment_anything import SamPredictor, sam_model_registry


from stable_processing.loader import load_dataset
from stable_processing.fake_masks import compute_box_from_mask, fake_logits_mask
from stable_processing.analysis import cluster_kmeans, apply_pca, overall_label, inter_group_cluster_kmeans, group_prototyping
from stable_processing.logging import print_with_color

from stable_processing.analysis import heatmap

import torch
import argparse
from tqdm import tqdm 

import numpy as np
import os 
K = 10
OVERALL_CLUSTER = 15
'''
    We need to determine which one of the label is the padding label and strange label
    Kind important
'''
class sam_batchify(SamPredictor):
    def __init__(self, sam_model: Sam) -> None:
        super().__init__(sam_model)
    
    def feature_extraction(self, images) -> torch.Tensor:
        
        '''
            Takes in BCHW images in torch CUDA and return encoder features
        '''
        
        return self.model.image_encoder(images)
    
    def mask_finegrainded_mask_generation(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        '''
            It will use the following methods
            mask input is a low resolution mask, (b, H, W)
            Where H=W=256 (how to transfer to something like this?)
        '''
        mask_T = masks.T
        boxes = compute_box_from_mask(mask_T).unsqueeze(0)
        mask_T = fake_logits_mask(mask_T).unsqueeze(0)
        features = features.unsqueeze(0)

        # we need to make masks and boxes in the shape of B4, and B1HW

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = None, 
            boxes = boxes,
            masks = mask_T
        ) # we do not have poitns and boxes as input, we only have a clustered masks

        image_position_embeddings = self.model.prompt_encoder.pe_layer((64, 64)).cuda().unsqueeze(0) # 
        
        # The input shape should be the following:
        low_res_result_masks, iou_prediction = self.model.mask_decoder(
            image_embeddings=features,  # B, C, 64, 64
            image_pe=image_position_embeddings, # B, C, 64, 64
            sparse_prompt_embeddings=sparse_embeddings, # B, 2, C
            dense_prompt_embeddings=dense_embeddings, # B, C, 64, 64
            multimask_output=False,
        )

        

        return low_res_result_masks, iou_prediction
    

    def point_fine_gradined_mask_generation(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        '''
            It will use the following methods
            mask input is a low resolution mask, (H, W)
            Where H=W=256 (how to transfer to something like this?)
        '''
        
        mask_T = masks.T
        points = torch.nonzero(mask_T == 1)+0.5 # move to the center of the pixel
        points_label = torch.ones(len(points)).unsqueeze(0)
        points = self.transform.apply_coords_torch(points, (64, 64)).unsqueeze(0)

        features = features.unsqueeze(0) 
        # how about encode all the points together?        

        sparse_input = (points, points_label)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points = sparse_input, 
            boxes = None,
            masks = None
        ) # we do not have poitns and boxes as input, we only have a clustered masks

        image_position_embeddings = self.model.prompt_encoder.pe_layer((64, 64)).cuda().unsqueeze(0) # 

        # The input shape should be the following:
        low_res_result_masks, iou_prediction = self.model.mask_decoder(
            image_embeddings=features,  # B, C, 64, 64
            image_pe=image_position_embeddings, # B, C, 64, 64
            sparse_prompt_embeddings=sparse_embeddings, # B, 2, C
            dense_prompt_embeddings=dense_embeddings, # B, C, 64, 64
            multimask_output=True,
        )
        ## we might need to merge multiple masks
        low_res_result_masks = low_res_result_masks.squeeze()
        merged_tensor, _ = torch.max(low_res_result_masks, dim=0)
        return merged_tensor, iou_prediction


def parser():
    parser = argparse.ArgumentParser("SAM Encoder Test", add_help=True)
    parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--debugging", type=str, default="False", help="if to save the internal output to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--batch_num", type=int, default=4, help="The number of images manipulating at one time, default=4")
    args = parser.parse_args()
    return args



def main():
    args = parser()
    sam_checkpoint = args.sam_checkpoint
    sam_version = args.sam_version
    
    print_with_color(f'Sam Checkpoint is :{sam_checkpoint}', 'YELLOW')
    print_with_color(f'Sam version is :{sam_version}', 'YELLOW')
    
    image_directory = args.image_dir
    output_directory = args.output_dir
    
    print_with_color(f'Image Directory is :{image_directory}', 'YELLOW')
    print_with_color(f'Output Directory is :{output_directory}', 'YELLOW')
    
    device = args.device
    
    debugging = (args.debugging == 'True')
    batch_number = args.batch_num
    
    if debugging:
        os.makedirs(os.path.join(output_directory, 'debugging'), exist_ok=True)
        debugging_dir = os.path.join(output_directory, 'debugging')
        print_with_color(f'The debugging mode is on, and the debugging result is stored in:{debugging_dir}', 'RED')
        print_with_color(f'To turn debugging mode off, simply set \"debugging False\"', 'RED')

    loader, image_number, padding_mask = load_dataset(image_directory, batch_size=batch_number, num_workers=2) 
    model = sam_batchify(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    features_saver = torch.zeros(size = (image_number, 256, 64, 64))
    batched_labels = torch.zeros(size = (image_number, 64, 64))
    batched_prototype = torch.zeros(size = (len(loader), K, 256))
    
    image_count = 0
    batch_count = 0
    
    with torch.no_grad():
        for images, names in tqdm(loader):
            images = images.to(device).squeeze(1)
            features = model.feature_extraction(images)

            features_saver[image_count:len(images)+image_count] = features.to('cpu')

            features = features.permute(0,2,3,1)       
                 
            down_sample_features = apply_pca(features)
            
            labels = cluster_kmeans(features=down_sample_features, n_clusters=K)
            
            labels = torch.as_tensor(labels, device='cuda')
            features = torch.as_tensor(features, device='cuda')
            
            prototype = group_prototyping(features, labels)
            
            
            batched_labels[image_count:len(images)+image_count] = labels.cpu()
            batched_prototype[batch_count] = prototype.cpu()
            
            image_count += len(images)
            batch_count += 1
    
    del model
    torch.cuda.empty_cache()

    print_with_color(f'Image encoding with hierachical clustering is accomplished' , 'GREEN')
    print_with_color(f'Saving Interal Result ...' , 'YELLOW')

    
    batched_prototype = batched_prototype.contiguous().cuda()
    prototype_clustered_result = inter_group_cluster_kmeans(batched_prototype, n_clusters=OVERALL_CLUSTER) # (prototype_len)
    prototype_clustered_result = prototype_clustered_result.reshape((len(loader), -1)) # (the number of batches, k)
    
    batched_labels = batched_labels.contiguous().cuda()            # (n//b, b, h, w)
    refined_lable = overall_label(batched_labels, prototype_clustered_result)

    # save labels and save features
    np.savez_compressed(f'{output_directory}/saved_labels', refined_lable.cpu().numpy())
    np.savez_compressed(f'{output_directory}/saved_features', features_saver.numpy())

    print_with_color(f'Interal result of clustering labels and features are saved at {output_directory}/saved_labels and {output_directory}/saved_features' , 'GREEN')

    del batched_labels, prototype_clustered_result, batched_prototype
    # We need to use refined labels as a masks and send it to sam mask encoder
    
    
    model = sam_batchify(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    
    file_name = sorted(os.listdir(image_directory))
    mask_dict = {}
    global_mask_index = {}

    # features_saver = torch.tensor(np.load('/home/planner/xiongbutian/ignores/output/saved_features.npz')['arr_0']).cuda()
    # refined_lable = torch.tensor(np.load('/home/planner/xiongbutian/ignores/output/saved_labels.npz')['arr_0']).cuda()


    
    # refined our mask using the decoding stratgy
    content = zip(features_saver, refined_lable, file_name)
    with torch.no_grad():
        for i, (features, masks, name) in tqdm(enumerate(content), total=len(features_saver), desc="Processing items"):
            features = features.cuda()
            masks = masks.cuda()
            
            masks_unique:torch.Tensor = torch.unique(masks) # we also need to remember the unique label, we need to use it for consistancy tracking
            refined_merged_masks = []
            filtered_label = []
            print_with_color(f' label name {masks_unique.shape}, features.shape {features.shape}, ', 'RED')
            for label in masks_unique:
                mask: torch.Tensor = (masks == label).float()
                # the rpompt point is actually at the padding region
                overlay = (mask*padding_mask).sum()
                if overlay >= 64:
                    continue
                filtered_label.append(label)
                refined_masks, _ = model.point_fine_gradined_mask_generation(mask, features)
                refined_masks = refined_masks
                print("refined_masks shape:", refined_masks.shape)
                refined_merged_masks.append(refined_masks.cpu())
                # exit()
                if debugging:
                    # if debugging, we will save logits as the internal result
                    debugging_name = name.split('.')[0]
                    heatmap(refined_masks, os.path.join(debugging_dir,f'{debugging_name}_{label}_logits.jpg'))
                    heatmap(mask, os.path.join(debugging_dir,f'{debugging_name}_{label}.jpg'))
            #mask_dict[name]  = torch.stack(refined_merged_masks).numpy()
            #global_mask_index[name]  = torch.stack(filtered_label).cpu().numpy()
            if len(refined_merged_masks) > 0:
                mask_dict[name]  = torch.stack(refined_merged_masks).numpy()
                global_mask_index[name]  = torch.stack(filtered_label).cpu().numpy()
            else:
                print_with_color(f'No valid masks found for {name}, skipping...', 'RED')

            
            # There is no unified shape since some of the unique label might be a lot, some might not be a unique label
        print_with_color('Saving refinement result ...', 'YELLOW')
        np.savez_compressed(f'{output_directory}/refined_mask.npz', **mask_dict)
        np.savez_compressed(f'{output_directory}/refined_label.npz', **global_mask_index)
    print_with_color(f'Mask refinement is accomplished, refined mask is saved at {output_directory}/refined_mask.npz', 'GREEN')

        
    

if __name__ == '__main__':
    main()    


# '''
#     We use the clustering mask as the input to the decoder
#     Therefore the latent we have used in batch should be preserve

#     Here is the overall stratgy
#     - We follow the clustering feature partsm we get the lattent, and we preserve it in some where
#     - We get the masks, and store it in some where, 
#     - We use the masks as the input and send it to the decoder model to gether with the features
# '''

# '''
#     We find that SAM cannot use dense mask as input along, we need to first convert it to bounding box
#     And then we fake the mask's logits as the input and convert masks to bounding box
# '''

# from segment_anything.modeling.sam import Sam
# from segment_anything import SamPredictor, sam_model_registry


# from stable_processing.loader import load_dataset
# from stable_processing.fake_masks import compute_box_from_mask, fake_logits_mask
# from stable_processing.analysis import cluster_kmeans, apply_pca, overall_label, inter_group_cluster_kmeans, group_prototyping
# from stable_processing.logging import print_with_color

# from stable_processing.analysis import heatmap

# import torch
# import argparse
# from tqdm import tqdm 

# import numpy as np
# import os 
# K = 10
# OVERALL_CLUSTER = 15
# '''
#     We need to determine which one of the label is the padding label and strange label
#     Kind important
# '''
# class sam_batchify(SamPredictor):
#     def __init__(self, sam_model: Sam) -> None:
#         super().__init__(sam_model)
    
#     def feature_extraction(self, images) -> torch.Tensor:
        
#         '''
#             Takes in BCHW images in torch CUDA and return encoder features
#         '''
        
#         return self.model.image_encoder(images)
    
#     def mask_finegrainded_mask_generation(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
#         '''
#             It will use the following methods
#             mask input is a low resolution mask, (b, H, W)
#             Where H=W=256 (how to transfer to something like this?)
#         '''
#         mask_T = masks.T
#         boxes = compute_box_from_mask(mask_T).unsqueeze(0)
#         mask_T = fake_logits_mask(mask_T).unsqueeze(0)
#         features = features.unsqueeze(0)

#         # we need to make masks and boxes in the shape of B4, and B1HW

#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points = None, 
#             boxes = boxes,
#             masks = mask_T
#         ) # we do not have poitns and boxes as input, we only have a clustered masks

#         image_position_embeddings = self.model.prompt_encoder.pe_layer((64, 64)).cuda().unsqueeze(0) # 
        
#         # The input shape should be the following:
#         low_res_result_masks, iou_prediction = self.model.mask_decoder(
#             image_embeddings=features,  # B, C, 64, 64
#             image_pe=image_position_embeddings, # B, C, 64, 64
#             sparse_prompt_embeddings=sparse_embeddings, # B, 2, C
#             dense_prompt_embeddings=dense_embeddings, # B, C, 64, 64
#             multimask_output=False,
#         )

        

#         return low_res_result_masks, iou_prediction
    

#     def point_fine_gradined_mask_generation(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

#         '''
#             It will use the following methods
#             mask input is a low resolution mask, (H, W)
#             Where H=W=256 (how to transfer to something like this?)
#         '''
        
#         mask_T = masks.T
#         points = torch.nonzero(mask_T == 1)+0.5 # move to the center of the pixel
#         points_label = torch.ones(len(points)).unsqueeze(0)
#         points = self.transform.apply_coords_torch(points, (64, 64)).unsqueeze(0)

#         features = features.unsqueeze(0) 
#         # how about encode all the points together?        

#         sparse_input = (points, points_label)

#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points = sparse_input, 
#             boxes = None,
#             masks = None
#         ) # we do not have poitns and boxes as input, we only have a clustered masks

#         image_position_embeddings = self.model.prompt_encoder.pe_layer((64, 64)).cuda().unsqueeze(0) # 

#         # The input shape should be the following:
#         low_res_result_masks, iou_prediction = self.model.mask_decoder(
#             image_embeddings=features,  # B, C, 64, 64
#             image_pe=image_position_embeddings, # B, C, 64, 64
#             sparse_prompt_embeddings=sparse_embeddings, # B, 2, C
#             dense_prompt_embeddings=dense_embeddings, # B, C, 64, 64
#             multimask_output=True,
#         )
#         ## we might need to merge multiple masks
#         low_res_result_masks = low_res_result_masks.squeeze()
#         merged_tensor, _ = torch.max(low_res_result_masks, dim=0)
#         return merged_tensor, iou_prediction


# def parser():
#     parser = argparse.ArgumentParser("SAM Encoder Test", add_help=True)
#     parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
#     parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
#     parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
#     parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
#     parser.add_argument("--debugging", type=str, default="False", help="if to save the internal output to image folder")
#     parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
#     parser.add_argument("--batch_num", type=int, default=4, help="The number of images manipulating at one time, default=4")
#     args = parser.parse_args()
#     return args



# def main():
#     args = parser()
#     sam_checkpoint = args.sam_checkpoint
#     sam_version = args.sam_version
    
#     print_with_color(f'Sam Checkpoint is :{sam_checkpoint}', 'YELLOW')
#     print_with_color(f'Sam version is :{sam_version}', 'YELLOW')
    
#     image_directory = args.image_dir
#     output_directory = args.output_dir
    
#     print_with_color(f'Image Directory is :{image_directory}', 'YELLOW')
#     print_with_color(f'Output Directory is :{output_directory}', 'YELLOW')
    
#     device = args.device
    
#     debugging = (args.debugging == 'True')
#     batch_number = args.batch_num
    
#     if debugging:
#         os.makedirs(os.path.join(output_directory, 'debugging'), exist_ok=True)
#         debugging_dir = os.path.join(output_directory, 'debugging')
#         print_with_color(f'The debugging mode is on, and the debugging result is stored in:{debugging_dir}', 'RED')
#         print_with_color(f'To turn debugging mode off, simply set \"debugging False\"', 'RED')

#     loader, image_number, padding_mask = load_dataset(image_directory, batch_size=batch_number, num_workers=2) 
#     model = sam_batchify(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
#     features_saver = torch.zeros(size = (image_number, 256, 64, 64))
#     batched_labels = torch.zeros(size = (image_number, 64, 64))
#     batched_prototype = torch.zeros(size = (len(loader), K, 256))
    
#     image_count = 0
#     batch_count = 0
    
#     with torch.no_grad():
#         for images, names in tqdm(loader):
#             images = images.to(device).squeeze(1)
#             features = model.feature_extraction(images)

#             features_saver[image_count:len(images)+image_count] = features.to('cpu')

#             features = features.permute(0,2,3,1)       
                 
#             down_sample_features = apply_pca(features)
            
#             labels = cluster_kmeans(features=down_sample_features, n_clusters=K)
            
#             labels = torch.as_tensor(labels, device='cuda')
#             features = torch.as_tensor(features, device='cuda')
            
#             prototype = group_prototyping(features, labels)
            
            
#             batched_labels[image_count:len(images)+image_count] = labels.cpu()
#             batched_prototype[batch_count] = prototype.cpu()
#             # print('batched_prototype is:\n',batched_prototype)
#             # print('batched_prototype shape is:', batched_prototype.shape)
            
#             image_count += len(images)
#             batch_count += 1
    
#     del model
#     torch.cuda.empty_cache()

#     print_with_color(f'Image encoding with hierachical clustering is accomplished' , 'GREEN')
#     print_with_color(f'Saving Interal Result ...' , 'YELLOW')

    
#     batched_prototype = batched_prototype.contiguous().cuda()
#     prototype_clustered_result = inter_group_cluster_kmeans(batched_prototype, n_clusters=OVERALL_CLUSTER) # (prototype_len)
#     prototype_clustered_result = prototype_clustered_result.reshape((len(loader), -1)) # (the number of batches, k)
    
#     batched_labels = batched_labels.contiguous().cuda()            # (n//b, b, h, w)
#     # print('batched_labels is:\n', batched_labels.shape)
#     refined_lable = overall_label(batched_labels, prototype_clustered_result)
#     # print('refined_labels is:\n', refined_lable.shape)

#     # save labels and save features
#     # np.savez_compressed(f'{output_directory}/saved_labels', refined_lable.cpu().numpy())
#     np.savez_compressed(f'{output_directory}/saved_features', features_saver.numpy())
#     np.savez_compressed(f'{output_directory}/saved_batch_labels', batched_prototype.cpu().numpy())

#     print_with_color(f'Interal result of clustering labels and features are saved at {output_directory}/saved_labels and {output_directory}/saved_features' , 'GREEN')

#     del batched_labels, prototype_clustered_result, batched_prototype
#     # We need to use refined labels as a masks and send it to sam mask encoder
    
    
#     model = sam_batchify(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    
#     file_name = sorted(os.listdir(image_directory))
#     mask_dict = {}
#     global_mask_index = {}

#     # features_saver = torch.tensor(np.load('/home/planner/xiongbutian/ignores/output/saved_features.npz')['arr_0']).cuda()
#     # refined_lable = torch.tensor(np.load('/home/planner/xiongbutian/ignores/output/saved_labels.npz')['arr_0']).cuda()


    
#     # refined our mask using the decoding stratgy
#     content = zip(features_saver, refined_lable, file_name)
#     with torch.no_grad():
#         for i, (features, masks, name) in tqdm(enumerate(content), total=len(features_saver), desc="Processing items"):
#             features = features.cuda()
#             masks = masks.cuda()
            
#             masks_unique = torch.unique(masks) # we also need to remember the unique label, we need to use it for consistancy tracking
#             refined_merged_masks = []
#             filtered_label = []
#             for label in masks_unique:
#                 mask = (masks == label).float()
#                 # the rpompt point is actually at the padding region
#                 overlay = (mask*padding_mask).sum()
#                 if overlay >= 64:
#                     continue
#                 filtered_label.append(label)
#                 refined_masks, _ = model.point_fine_gradined_mask_generation(mask, features)
#                 refined_masks = refined_masks
#                 refined_merged_masks.append(refined_masks.cpu())
#                 if debugging:
#                     # if debugging, we will save logits as the internal result
#                     debugging_name = name.split('.')[0]
#                     heatmap(refined_masks, os.path.join(debugging_dir,f'{debugging_name}_{label}_logits.jpg'))
#                     heatmap(mask, os.path.join(debugging_dir,f'{debugging_name}_{label}.jpg'))
#             mask_dict[name]  = torch.stack(refined_merged_masks).numpy()
#             global_mask_index[name]  = torch.stack(filtered_label).cpu().numpy()
#             # There is no unified shape since some of the unique label might be a lot, some might not be a unique label
#         print_with_color('Saving refinement result ...', 'YELLOW')
#         np.savez_compressed(f'{output_directory}/refined_mask.npz', **mask_dict)
#         np.savez_compressed(f'{output_directory}/refined_label.npz', **global_mask_index)
#     print_with_color(f'Mask refinement is accomplished, refined mask is saved at {output_directory}/refined_mask.npz', 'GREEN')

        
    

# if __name__ == '__main__':
#     main()    


# from segment_anything.modeling.sam import Sam
# from segment_anything import SamPredictor, sam_model_registry
# from stable_processing.loader import load_dataset
# from stable_processing.fake_masks import compute_box_from_mask, fake_logits_mask
# from stable_processing.analysis import cluster_kmeans, apply_pca, overall_label, inter_group_cluster_kmeans, group_prototyping
# from stable_processing.logging import print_with_color
# from stable_processing.analysis import heatmap

# import torch
# import argparse
# from tqdm import tqdm
# import numpy as np
# import os

# K = 10
# OVERALL_CLUSTER = 15

# class sam_batchify(SamPredictor):
#     def __init__(self, sam_model: Sam) -> None:
#         super().__init__(sam_model)

#     def feature_extraction(self, images) -> torch.Tensor:
#         return self.model.image_encoder(images)

#     def mask_finegrainded_mask_generation(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
#         mask_T = masks.T
#         boxes = compute_box_from_mask(mask_T).unsqueeze(0)
#         mask_T = fake_logits_mask(mask_T).unsqueeze(0)
#         features = features.unsqueeze(0)

#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=None,
#             boxes=boxes,
#             masks=mask_T
#         )

#         image_position_embeddings = self.model.prompt_encoder.pe_layer((64, 64)).cuda().unsqueeze(0)

#         low_res_result_masks, iou_prediction = self.model.mask_decoder(
#             image_embeddings=features,
#             image_pe=image_position_embeddings,
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )

#         return low_res_result_masks, iou_prediction

#     def point_fine_gradined_mask_generation(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
#         mask_T = masks.T
#         points = torch.nonzero(mask_T == 1) + 0.5
#         points_label = torch.ones(len(points)).unsqueeze(0)
#         points = self.transform.apply_coords_torch(points, (64, 64)).unsqueeze(0)

#         features = features.unsqueeze(0)
#         sparse_input = (points, points_label)

#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=sparse_input,
#             boxes=None,
#             masks=None
#         )

#         image_position_embeddings = self.model.prompt_encoder.pe_layer((64, 64)).cuda().unsqueeze(0)

#         low_res_result_masks, iou_prediction = self.model.mask_decoder(
#             image_embeddings=features,
#             image_pe=image_position_embeddings,
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=True,
#         )

#         low_res_result_masks = low_res_result_masks.squeeze()
#         merged_tensor, _ = torch.max(low_res_result_masks, dim=0)
#         return merged_tensor, iou_prediction

# def parser():
#     parser = argparse.ArgumentParser("SAM Encoder Test", add_help=True)
#     parser.add_argument("--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h")
#     parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
#     parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
#     parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
#     parser.add_argument("--debugging", type=str, default="False", help="if to save the internal output to image folder")
#     parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
#     parser.add_argument("--batch_num", type=int, default=4, help="The number of images manipulating at one time, default=4")
#     args = parser.parse_args()
#     return args

# # def other_mask_generation(mask_list):
# #     for mask in mask_list:
        
    
    
# def main():
#     args = parser()
#     sam_checkpoint = args.sam_checkpoint
#     sam_version = args.sam_version

#     print_with_color(f'Sam Checkpoint is :{sam_checkpoint}', 'YELLOW')
#     print_with_color(f'Sam version is :{sam_version}', 'YELLOW')

#     image_directory = args.image_dir
#     output_directory = args.output_dir

#     print_with_color(f'Image Directory is :{image_directory}', 'YELLOW')
#     print_with_color(f'Output Directory is :{output_directory}', 'YELLOW')

#     device = args.device
#     debugging = (args.debugging == 'True')
#     batch_number = args.batch_num

#     if debugging:
#         os.makedirs(os.path.join(output_directory, 'debugging'), exist_ok=True)
#         debugging_dir = os.path.join(output_directory, 'debugging')
#         print_with_color(f'The debugging mode is on, and the debugging result is stored in:{debugging_dir}', 'RED')
#         print_with_color(f'To turn debugging mode off, simply set "debugging False"', 'RED')

#     loader, image_number, padding_mask = load_dataset(image_directory, batch_size=batch_number, num_workers=2)
#     model = sam_batchify(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
#     features_saver = torch.zeros(size=(image_number, 256, 64, 64))
#     batched_labels = torch.zeros(size=(image_number, 64, 64))
#     batched_prototype = torch.zeros(size=(len(loader), K, 256))

#     image_count = 0
#     batch_count = 0

#     with torch.no_grad():
#         for images, names in tqdm(loader):
#             images = images.to(device).squeeze(1)
#             features = model.feature_extraction(images)

#             features_saver[image_count:len(images)+image_count] = features.to('cpu')

#             features = features.permute(0, 2, 3, 1)

#             down_sample_features = apply_pca(features)

#             labels = cluster_kmeans(features=down_sample_features, n_clusters=K)

#             labels = torch.as_tensor(labels, device='cuda')
#             features = torch.as_tensor(features, device='cuda')

#             prototype = group_prototyping(features, labels)

#             batched_labels[image_count:len(images)+image_count] = labels.cpu()
#             batched_prototype[batch_count] = prototype.cpu()

#             image_count += len(images)
#             batch_count += 1

#     del model
#     torch.cuda.empty_cache()

#     print_with_color(f'Image encoding with hierarchical clustering is accomplished', 'GREEN')
#     print_with_color(f'Saving Internal Result ...', 'YELLOW')

#     batched_prototype = batched_prototype.contiguous().cuda()
#     prototype_clustered_result = inter_group_cluster_kmeans(batched_prototype, n_clusters=OVERALL_CLUSTER)
#     prototype_clustered_result = prototype_clustered_result.reshape((len(loader), -1))

#     batched_labels = batched_labels.contiguous().cuda()
#     refined_lable = overall_label(batched_labels, prototype_clustered_result)

#     np.savez_compressed(f'{output_directory}/saved_features', features_saver.numpy())
#     np.savez_compressed(f'{output_directory}/saved_batch_labels', batched_prototype.cpu().numpy())

#     print_with_color(f'Internal result of clustering labels and features are saved at {output_directory}/saved_features and {output_directory}/saved_batch_labels', 'GREEN')

#     del batched_labels, prototype_clustered_result, batched_prototype

#     model = sam_batchify(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

#     file_names = sorted(os.listdir(image_directory))
#     mask_dict = {}
#     global_mask_index = {}

#     content = zip(features_saver, refined_lable, file_names)
#     with torch.no_grad():
#         for i, (features, masks, name) in tqdm(enumerate(content), total=len(features_saver), desc="Processing items"):
#             features = features.cuda()
#             masks = masks.cuda()

#             masks_unique = torch.unique(masks)
#             refined_merged_masks = []
#             filtered_label = []
#             for label in masks_unique:
#                 mask = (masks == label).float()
#                 overlay = (mask * padding_mask).sum()
#                 if overlay >= 64:
#                     continue
#                 filtered_label.append(label)
#                 refined_masks, _ = model.point_fine_gradined_mask_generation(mask, features)
#                 refined_masks = refined_masks
#                 refined_merged_masks.append(refined_masks.cpu())
#                 if debugging:
#                     debugging_name = name.split('.')[0]
#                     heatmap(refined_masks, os.path.join(debugging_dir, f'{debugging_name}_{label}_logits.jpg'))
#                     heatmap(mask, os.path.join(debugging_dir, f'{debugging_name}_{label}.jpg'))
            
#             # other_mask = 

#             # Create subfolder for each image group
#             subfolder = os.path.join(output_directory, os.path.splitext(name)[0])
#             os.makedirs(subfolder, exist_ok=True)

#             np.savez_compressed(os.path.join(subfolder, 'refined_mask.npz'), **{name: torch.stack(refined_merged_masks).numpy()})
#             np.savez_compressed(os.path.join(subfolder, 'refined_label.npz'), **{name: torch.stack(filtered_label).cpu().numpy()})

#     print_with_color(f'Mask refinement is accomplished, refined masks are saved in {output_directory}', 'GREEN')

# if __name__ == '__main__':
#     main()
