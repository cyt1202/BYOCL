IMAGE_PATH=$1
OUTPUT_DIR=$2

CUDA_VISIBLE_DEVICES=7 python -W ignore new_fine_mask_segmentation.py \
    --sam_checkpoint /home/xiongbutian/workspace/masa-main/saved_models/pretrain_weights/sam_vit_h_4b8939.pth \
    --image_dir $IMAGE_PATH \
    --batch_num 4 \
    --output_dir $OUTPUT_DIR \
    --debugging False \
    --device cuda

python visualization_refined.py \
    -i  $IMAGE_PATH\
    -m $OUTPUT_DIR/refined_mask.npz \
    -o $OUTPUT_DIR
