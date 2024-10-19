# CUDA_VISIBLE_DEVICES=2 python -W ignore clustering_features.py \
#     --sam_checkpoint /home/planner/xiongbutian/Foundation_Models/SAM/sam_vit_h_4b8939.pth \
#     --image_dir /home/planner/xiongbutian/sc_latent_sam/images \
#     --batch_num 8 \
#     --output_dir output \
#     --debugging True \
#     --device cuda 

# python visualization.py \
#     -i /data/grocery_store/10F/input/ \
#     -m output/saved_labels.npz \
#     -o output/ 
CUDA_VISIBLE_DEVICES=2 python -W ignore clustering_features.py \
    --sam_checkpoint /home/planner/xiongbutian/Foundation_Models/SAM/sam_vit_h_4b8939.pth \
    --image_dir /home/xiongbutian/workspace/Sam2/segment-anything-2/valid/JPEGImages/0d0030a7\
    --batch_num 8 \
    --output_dir output \
    --debugging True \
    --device cuda 

python visualization.py \
    -i /home/xiongbutian/workspace/Sam2/segment-anything-2/valid/JPEGImages/0d0030a7 \
    -m output/saved_labels.npz \
    -o output/ 

