#!/bin/bash

# 创建结果目录
RESULTS_DIR="grid_resu"
mkdir -p "$RESULTS_DIR"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_FILE="$RESULTS_DIR/grid_search_${TIMESTAMP}.csv"

# 写入CSV表头
echo "M,MaxLayer,EFC,EFS,HammingThreshold,Recall,AvgQueryTime_ms,DistOpsPerQuery,BuildTime_ms,QPS" > "$CSV_FILE"

# 定义参数网格
M_VALUES=(32 40 57 64 80 128)
MAX_LAYER=7
EFC_VALUES=(2000)
# 之前是一个固定值，改为：下界、上界、步长
EFS_LOWER=128
EFS_UPPER=1024
EFS_STEP=16
HAMMING_THRESHOLD=(40 41 42 43)  # optional list; values <=0 will not pass the flag

# 生成 EFS_VALUES 基于下界/上界/步长（包含校验）
EFS_VALUES=()
if [ "$EFS_STEP" -le 0 ]; then
    echo "Error: EFS_STEP must be > 0"
    exit 1
fi
if [ "$EFS_LOWER" -gt "$EFS_UPPER" ]; then
    echo "Error: EFS_LOWER must be <= EFS_UPPER"
    exit 1
fi

for (( val=EFS_LOWER; val<=EFS_UPPER; val+=EFS_STEP )); do
    EFS_VALUES+=( "$val" )
done

# 进度计数 - 包含 HAMMING_THRESHOLD 维度
TOTAL_RUNS=$((${#M_VALUES[@]} * ${#EFC_VALUES[@]} * ${#EFS_VALUES[@]} * ${#HAMMING_THRESHOLD[@]}))
CURRENT_RUN=0

echo "=========================================="
echo "Starting Grid Search with $TOTAL_RUNS configurations (hamming_threshold=${HAMMING_THRESHOLD[@]})"
echo "Results will be saved to: $CSV_FILE"
echo "=========================================="
echo ""

# 遍历参数组合
for M in "${M_VALUES[@]}"; do
    for EFC in "${EFC_VALUES[@]}"; do
        for EFS in "${EFS_VALUES[@]}"; do
            for HT in "${HAMMING_THRESHOLD[@]}"; do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                
                echo "----------------------------------------"
                echo "Run $CURRENT_RUN/$TOTAL_RUNS: M=$M, MaxLayer=$MAX_LAYER, EFC=$EFC, EFS=$EFS, HAMMING=$HT"
                echo "----------------------------------------"
                
                EXTRA_FLAGS=""
                if [ "$HT" -gt 0 ]; then
                    EXTRA_FLAGS="$EXTRA_FLAGS --hamming_threshold $HT"
                fi

                # 运行程序并捕获输出
                OUTPUT=$(./hng3 --m $M --max_layer $MAX_LAYER --efc $EFC --efs $EFS $EXTRA_FLAGS 2>&1)
                
                # 提取关键指标
                RECALL=$(echo "$OUTPUT" | grep "Average recall@10:" | awk '{print $3}')
                AVG_QUERY_TIME=$(echo "$OUTPUT" | grep "Average query time:" | awk '{print $4}')
                DIST_OPS=$(echo "$OUTPUT" | grep "Average distance ops per query:" | awk '{print $6}')
                BUILD_TIME=$(echo "$OUTPUT" | grep "Index build time:" | awk '{print $4}')
                QPS=$(echo "$OUTPUT" | grep "Queries per second:" | awk '{print $4}')
                
                # 处理可能的空值
                RECALL=${RECALL:-"N/A"}
                AVG_QUERY_TIME=${AVG_QUERY_TIME:-"N/A"}
                DIST_OPS=${DIST_OPS:-"N/A"}
                BUILD_TIME=${BUILD_TIME:-"N/A"}
                QPS=${QPS:-"N/A"}
                
                # 写入CSV (新增 HammingThreshold 列)
                echo "$M,$MAX_LAYER,$EFC,$EFS,$HT,$RECALL,$AVG_QUERY_TIME,$DIST_OPS,$BUILD_TIME,$QPS" >> "$CSV_FILE"
                
                # 打印当前结果摘要
                echo "  Recall: $RECALL"
                echo "  Avg Query Time: $AVG_QUERY_TIME ms"
                echo "  Dist Ops/Query: $DIST_OPS"
                echo "  Build Time: $BUILD_TIME ms"
                echo "  QPS: $QPS"
                echo ""
                
                # 短暂延迟，避免系统过载
                sleep 1
            done
        done
    done
done

echo "=========================================="
echo "Grid Search Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $CSV_FILE"
echo ""

# 生成结果摘要 - 更新字段索引（HammingThreshold 在 CSV 的第5列）
echo "Top 5 configurations by Recall:"
echo "-------------------------------"
tail -n +2 "$CSV_FILE" | sort -t',' -k6 -rn | head -5 | \
    awk -F',' '{printf "M=%s EFC=%s EFS=%s HAM=%s: Recall=%.6f, QTime=%.3fms, DistOps=%.0f\n", $1, $3, $4, $5, $6, $7, $8}'
echo ""

echo "Top 5 configurations by Query Speed (lowest time):"
echo "--------------------------------------------------"
tail -n +2 "$CSV_FILE" | sort -t',' -k7 -n | head -5 | \
    awk -F',' '{printf "M=%s EFC=%s EFS=%s HAM=%s: QTime=%.3fms, Recall=%.6f, DistOps=%.0f\n", $1, $3, $4, $5, $7, $6, $8}'
echo ""

echo "Top 5 configurations by Distance Efficiency (lowest ops):"
echo "---------------------------------------------------------"
tail -n +2 "$CSV_FILE" | sort -t',' -k8 -n | head -5 | \
    awk -F',' '{printf "M=%s EFC=%s EFS=%s HAM=%s: DistOps=%.0f, Recall=%.6f, QTime=%.3fms\n", $1, $3, $4, $5, $8, $6, $7}'
echo ""