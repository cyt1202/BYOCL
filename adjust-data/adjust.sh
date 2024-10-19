folderA="/home/xiongbutian/workspace/sc_latent_sam/output_files/Mose-script-output"
folderB="/home/xiongbutian/workspace/sc_latent_sam/Annotations/Morse"
#outputBaseDir="/home/xiongbutian/workspace/sc_latent_sam/output_files/Davis-script-adjust-output"

outputBaseDir="/home/xiongbutian/workspace/sc_latent_sam/output_files/Mose-script-adjust"

# 遍历文件夹A下的所有子文件夹
for dir in "$folderA"/*/; do
    # 去掉目录路径的末尾斜杠
    dirName=$(basename "$dir")

    # 如果是目录
    if [ -d "$dir" ]; then
        echo "处理目录: $dir"

        # 创建以子文件夹名字命名的输出文件夹
        outputDir="$outputBaseDir/$dirName"
        #mkdir -p "$outputDir"

        # 获取文件A中名为 refined_mask.npz 的文件
        fileA="$dir/refined_mask.npz"
        if [ -f "$fileA" ]; then
            echo "找到文件A: $fileA"
        else
            echo "文件A: $fileA 不存在，跳过..."
            continue
        fi

        # 获取文件B中与 dirName 名字相同的文件
        fileB="$folderB/$dirName.npz"
        if [ -f "$fileB" ]; then
            echo "找到文件B: $fileB"
        else
            echo "文件B: $fileB 不存在，跳过..."
            continue
        fi

        # 执行 Python 脚本，处理文件A和文件B
        CUDA_VISIBLE_DEVICES=4 python -W ignore adjust-size-01.py \
            --data "$fileA" \
            --gt "$fileB" \
            --output "$outputDir"

    fi
done


