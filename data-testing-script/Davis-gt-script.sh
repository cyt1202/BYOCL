#!/bin/bash

# 定义文件夹A的路径
folderA="/home/xiongbutian/workspace/Sam2/segment-anything-2/train/Annotations"
outputBaseDir="/home/xiongbutian/workspace/sc_latent_sam/Annotations/Morse"

# 遍历文件夹A下的所有子文件夹
for dir in "$folderA"/*/; do
    # 去掉目录路径的末尾斜杠
    dirName=$(basename "$dir")

    # 如果是目录
    if [ -d "$dir" ]; then
        echo "处理目录: $dir"

        # 创建以子文件夹名字命名的输出文件夹
        outputDir="$outputBaseDir/$dirName"
        # mkdir -p "$outputDir"

        # 执行 Python 脚本
        python matrice_evaluation1.py \
            -g  "$dir" \
            -o "$outputDir"
        # CUDA_VISIBLE_DEVICES=4 python -W ignore fine_mask_segmentation1.py \
        #     --sam_checkpoint /home/xiongbutian/workspace/masa-main/saved_models/pretrain_weights/sam_vit_h_4b8939.pth \
        #     --image_dir "$dir" \
        #     --batch_num 4 \
        #     --output_dir "$outputDir" \
        #     --debugging False \
        #     --device cuda
    fi
done
