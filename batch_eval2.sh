##!/bin/bash
#
#export CUDA_VISIBLE_DEVICES='0,1'
#CONFIG="/home/ysz/mmaction2-main/mmaction2-main/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb.py"
#CHECKPOINT_DIR="/nfs/ysz/uniformerv2_result/fire/base/base"
##OUTPUT_CSV="${CHECKPOINT_DIR}/eval_results.csv"
#GPUS=2
#
##echo "checkpoint,top1_acc" > $OUTPUT_CSV
#
#index=11
#
#for CKPT_PATH in $(find $CHECKPOINT_DIR -name "epoch_*.pth" | sort -V); do
##    echo "Evaluating checkpoint: ${CKPT_PATH}"
#    if [ -e "$CKPT_PATH" ]; then
#          echo "Found file: $CKPT_PATH" #PORT=29501
#          bash /home/ysz/mmaction2-main/mmaction2-main/tools/dist_test.sh $CONFIG $CKPT_PATH $GPUS \
#          --dump /nfs/ysz/uniformerv2_result/fire/base/base/results/epoch_$index.pkl
#          ((index++))
#    else
#      echo "No files matching 'epoch_*.pth'found in $CHECKPOINT_DIR"
#      break
#    fi
##    TOP1=$(echo "$RESULT" | grep -oP 'accuracy/top1:\s*\K[0-9.]+' | head -n 1)
##    echo "$(basename $CKPT_PATH),${TOP1}" >> $OUTPUT_CSV
#done

#!/bin/bash

export CUDA_VISIBLE_DEVICES='0,1'
CONFIG="/home/ysz/mmaction2-main/mmaction2-main/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_8xb32-u8_kinetics400-rgb.py"
CHECKPOINT_DIR="/nfs/ysz/uniformerv2_result/fire2/ablation/only_4"
OUTPUT_CSV="${CHECKPOINT_DIR}/eval_results.csv"
GPUS=2

# 设置起始和结束epoch (可根据需要修改)
START_EPOCH=4
END_EPOCH=4

# 创建CSV文件并写入表头
echo "epoch,top1_acc,top5_acc,mean_average_precision" > $OUTPUT_CSV

# 获取所有checkpoint文件并按数字顺序排序
checkpoints=($(find $CHECKPOINT_DIR -name "epoch_*.pth" | sort -V))
echo "找到 ${#checkpoints[@]} 个checkpoint文件"

# 遍历所有checkpoint文件
for CKPT_PATH in "${checkpoints[@]}"; do
    # 从文件名中提取epoch号
    epoch_num=$(basename "$CKPT_PATH" | grep -oP 'epoch_\K[0-9]+' | head -1)

    # 检查epoch是否在指定范围内
    if [ "$epoch_num" -lt "$START_EPOCH" ] || [ "$epoch_num" -gt "$END_EPOCH" ]; then
        echo "跳过 epoch ${epoch_num} (不在范围 ${START_EPOCH}-${END_EPOCH} 内)"
        continue
    fi

    echo "处理 checkpoint: ${CKPT_PATH}, epoch: ${epoch_num}"

    # 创建临时日志文件
    LOG_FILE="${CHECKPOINT_DIR}/temp_epoch_${epoch_num}.log"

    # 运行测试并捕获输出到日志文件
    echo "开始测试 epoch ${epoch_num}..."
    bash /home/ysz/mmaction2-main/mmaction2-main/tools/dist_test.sh $CONFIG $CKPT_PATH $GPUS \
        --dump /nfs/ysz/uniformerv2_result/fire2/ablation/only_4/results/epoch_${epoch_num}.pkl 2>&1 | tee $LOG_FILE

    # 从日志中提取指标 - 根据实际日志格式调整
    # 使用更精确的模式匹配来提取指标
    TOP1=$(grep -oP "acc/top1:\s*\K[0-9.]+" $LOG_FILE | tail -1 || echo "")
    TOP5=$(grep -oP "acc/top5:\s*\K[0-9.]+" $LOG_FILE | tail -1 || echo "")
    MAP=$(grep -oP "acc/mean_average_precision:\s*\K[0-9.]+" $LOG_FILE | tail -1 || echo "")

    # 如果没有找到任何指标，全部设为空
    if [ -z "$TOP1" ] && [ -z "$TOP5" ] && [ -z "$MAP" ]; then
        TOP1=""
        TOP5=""
        MAP=""
        echo "警告: 未找到 epoch ${epoch_num} 的评估指标"
    else
        echo "epoch ${epoch_num} 结果: top1=${TOP1}, top5=${TOP5}, mAP=${MAP}"
    fi

    # 将结果写入CSV
    echo "${epoch_num},${TOP1},${TOP5},${MAP}" >> $OUTPUT_CSV

    # 删除临时日志文件
    rm -f $LOG_FILE

    echo "完成 epoch ${epoch_num} 的测试"
    echo "----------------------------------------"
done

echo "所有测试完成，结果保存在 ${OUTPUT_CSV}"