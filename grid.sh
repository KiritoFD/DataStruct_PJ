#!/bin/bash

# 创建结果目录
RESULTS_DIR="grid_results"
mkdir -p "$RESULTS_DIR"

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CSV_FILE="$RESULTS_DIR/grid_search_${TIMESTAMP}.csv"

# 写入CSV表头
echo "M,MaxLayer,EFC,EFS,Recall,AvgQueryTime_ms,DistOpsPerQuery,BuildTime_ms,QPS" > "$CSV_FILE"

# 定义参数网格
M_VALUES=(32 40 64 80)
MAX_LAYER=7
EFC_VALUES=(600 1000 1300)
# 之前是一个固定值，改为组合
EFS_VALUES=(100 250 500 600 800 1000)

# 进度计数
TOTAL_RUNS=$((${#M_VALUES[@]} * ${#EFC_VALUES[@]} * ${#EFS_VALUES[@]}))
CURRENT_RUN=0

echo "=========================================="
echo "Starting Grid Search with $TOTAL_RUNS configurations"
echo "Results will be saved to: $CSV_FILE"
echo "=========================================="
echo ""

# 遍历参数组合
for M in "${M_VALUES[@]}"; do
    for EFC in "${EFC_VALUES[@]}"; do
        for EFS in "${EFS_VALUES[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            
            echo "----------------------------------------"
            echo "Run $CURRENT_RUN/$TOTAL_RUNS: M=$M, MaxLayer=$MAX_LAYER, EFC=$EFC, EFS=$EFS"
            echo "----------------------------------------"
            
            # 运行程序并捕获输出
            OUTPUT=$(./hng2 --m $M --max_layer $MAX_LAYER --efc $EFC --efs $EFS 2>&1)
            
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
            
            # 写入CSV
            echo "$M,$MAX_LAYER,$EFC,$EFS,$RECALL,$AVG_QUERY_TIME,$DIST_OPS,$BUILD_TIME,$QPS" >> "$CSV_FILE"
            
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

echo "=========================================="
echo "Grid Search Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $CSV_FILE"
echo ""

# 生成结果摘要
echo "Top 5 configurations by Recall:"
echo "-------------------------------"
tail -n +2 "$CSV_FILE" | sort -t',' -k5 -rn | head -5 | \
    awk -F',' '{printf "M=%s EFC=%s EFS=%s: Recall=%.6f, QTime=%.3fms, DistOps=%.0f\n", $1, $3, $4, $5, $6, $7}'
echo ""

echo "Top 5 configurations by Query Speed (lowest time):"
echo "--------------------------------------------------"
tail -n +2 "$CSV_FILE" | sort -t',' -k6 -n | head -5 | \
    awk -F',' '{printf "M=%s EFC=%s EFS=%s: QTime=%.3fms, Recall=%.6f, DistOps=%.0f\n", $1, $3, $4, $6, $5, $7}'
echo ""

echo "Top 5 configurations by Distance Efficiency (lowest ops):"
echo "---------------------------------------------------------"
tail -n +2 "$CSV_FILE" | sort -t',' -k7 -n | head -5 | \
    awk -F',' '{printf "M=%s EFC=%s EFS=%s: DistOps=%.0f, Recall=%.6f, QTime=%.3fms\n", $1, $3, $4, $7, $5, $6}'
echo ""