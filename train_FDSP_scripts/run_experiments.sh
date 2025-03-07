#!/bin/bash

# 定义测试参数
BATCH_SIZES=(2 4 8)  # 测试不同的 batch size
GRADIENT_ACCUMULATION_STEPS=(8)  # 测试不同的梯度累积步数
CPU_OFFLOADS=(true false)  # 是否启用 CPUOffload
AUTO_WRAP_POLICIES=(true false)  # 是否启用 auto_wrap_policy

# 输出结果文件
OUTPUT_FILE="experiment_results.csv"
echo "Batch Size,Gradient Accumulation Steps,CPU Offload,Auto Wrap Policy,Average Throughput,Tensor Memory Usage" > $OUTPUT_FILE

# 运行实验
for batch_size in "${BATCH_SIZES[@]}"; do
    for grad_acc_steps in "${GRADIENT_ACCUMULATION_STEPS[@]}"; do
        for cpu_offload in "${CPU_OFFLOADS[@]}"; do
            for auto_wrap in "${AUTO_WRAP_POLICIES[@]}"; do
                # 设置环境变量
                export BATCH_SIZE=$batch_size
                export GRADIENT_ACCUMULATION_STEPS=$grad_acc_steps
                export CPU_OFFLOAD=$cpu_offload
                export AUTO_WRAP_POLICY=$auto_wrap

                # 清理之前的日志文件
                rm -f throughput.log memory.log

                # 启动分布式训练
                torchrun --nproc_per_node=8 train_fairScale_FDSP.py --batch_size $batch_size --gradient_accumulation_steps $grad_acc_steps

                # 提取吞吐量数据
                throughput_values=$(cat throughput.log)

                # 计算行数
                total_lines=$(echo "$throughput_values" | wc -l)

                # 计算中间行的起始和结束位置
                start_line=$(( (total_lines - 10) / 2 + 1 ))
                end_line=$(( start_line + 9 ))

                # 提取中间 10 行
                middle_values=$(echo "$throughput_values" | tail -n +$start_line | head -n 10)

                # 计算中间 10 行的平均值
                average_throughput=$(echo "$middle_values" | awk '{sum+=$1; count++} END {print sum/count}')
                if [ -z "$average_throughput" ]; then
                    average_throughput="N/A"
                fi

                # 输出结果
                echo "Average Throughput (middle 10 values): $average_throughput"

                # 处理 memory.log 文件
                echo "Processing memory.log..."
                memory_values=$(cat memory.log)
                total_memory_lines=$(echo "$memory_values" | wc -l)

                # 计算中间10次的起始和结束行号
                start_memory_line=$(( (total_memory_lines - 10) / 2 + 1 ))
                end_memory_line=$(( start_memory_line + 9 ))

                # 提取中间10次的显存开销数据
                middle_memory_values=$(echo "$memory_values" | tail -n +$start_memory_line | head -n 10)

                # 计算中间10次的平均显存开销
                average_memory=$(echo "$middle_memory_values" | awk '{sum+=$1; count++} END {print sum/count}')
                if [ -z "$average_memory" ]; then
                    average_memory="N/A"
                fi

                echo "Average Memory Usage (middle 10 values): $average_memory MB"

                # 记录结果
                echo "$batch_size,$grad_acc_steps,$cpu_offload,$auto_wrap,$average_throughput,$average_memory" >> $OUTPUT_FILE
            done
        done
    done
done

echo "Experiment completed. Results saved in $OUTPUT_FILE"