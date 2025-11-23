import torch
import numpy as np
import json
import sys
import os
from tqdm import tqdm

# --- 配置参数 ---
INPUT_FILE = "data_o/sift/base.txt"
CACHE_FILE = "data_o/sift/base.bin"
OUTPUT_FILE = "results.json"
DIMENSION = 128
TOP_K = 10
BATCH_SIZE = 512               # 8G显存：512 * 100W * 2bytes(fp16) ≈ 1GB 查询批次占用
DATA_SHARD_SIZE = 500000       # 数据库分片大小：500K * 128 * 2bytes ≈ 128MB

def load_data(filepath, cache_path=None):
    """读取txt或从缓存读取，返回 float32 的 numpy array"""
    
    # 优先检查缓存
    if cache_path and os.path.exists(cache_path):
        print(f"检测到缓存文件: {cache_path}，正在加载...")
        try:
            data = np.load(cache_path)
            print(f"缓存加载成功: {data.shape[0]} 行, 维度 {data.shape[1]}")
            return data.astype('float32')
        except Exception as e:
            print(f"警告: 缓存加载失败 ({e})，将从源文件重新读取")
    
    # 缓存不存在或加载失败，从源文件读取
    print(f"正在读取数据: {filepath} ...")
    data_list = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading"):
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) != DIMENSION:
                    continue
                data_list.append([float(x) for x in parts])
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
        sys.exit(1)
    
    data = np.array(data_list).astype('float32')
    
    # 保存缓存供下次使用
    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        try:
            np.save(cache_path, data)
            print(f"已保存缓存: {cache_path}")
        except Exception as e:
            print(f"警告: 缓存保存失败 ({e})")
    
    return data

def main():
    # 1. 检查 GPU
    if not torch.cuda.is_available():
        print("错误: 未检测到 GPU，请安装 PyTorch CUDA 版本")
        sys.exit(1)
    
    device = torch.device("cuda")
    print(f"使用设备: {torch.cuda.get_device_name(0)}")

    # 2. 加载数据 (优先使用缓存)
    numpy_data = load_data(INPUT_FILE, cache_path=CACHE_FILE)
    total_vectors = numpy_data.shape[0]
    
    if total_vectors == 0:
        print("数据为空")
        sys.exit(1)

    print(f"数据加载完成: {total_vectors} 行, 维度 {DIMENSION}")
    print("正在将数据搬运至 GPU 并转换为 FP16 (半精度)...")

    data_gpu = torch.from_numpy(numpy_data).to(device).half()

    print("预计算向量模长...")
    data_norms = (data_gpu ** 2).sum(dim=1, keepdim=True)
    data_norms_t = data_norms.t()

    # --- 新增：数据库分片优化 (为8G显存) ---
    num_data_shards = (total_vectors + DATA_SHARD_SIZE - 1) // DATA_SHARD_SIZE
    data_shards = []
    data_norms_shards = []
    
    print(f"正在分片数据库 (共 {num_data_shards} 片，每片 {DATA_SHARD_SIZE} 向量)...")
    for shard_idx in range(num_data_shards):
        shard_start = shard_idx * DATA_SHARD_SIZE
        shard_end = min((shard_idx + 1) * DATA_SHARD_SIZE, total_vectors)
        data_shards.append(data_gpu[shard_start:shard_end])
        data_norms_shards.append(data_norms[shard_start:shard_end])

    final_indices_tensors = []
    final_dists_tensors = []

    print(f"开始 PyTorch 暴力搜索 (Top-{TOP_K})...")
    num_batches = (total_vectors + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Searching (GPU Running Continuously)"):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, total_vectors)
            
            query_batch = data_gpu[batch_start:batch_end]
            query_norms = data_norms[batch_start:batch_end]

            # --- 新增：对每个数据库分片分别计算，避免显存溢出 ---
            batch_topk_indices = None
            batch_topk_dists = None

            for shard_idx in range(num_data_shards):
                shard_data = data_shards[shard_idx]
                shard_norms = data_norms_shards[shard_idx]
                shard_norms_t = shard_norms.t()
                shard_start_global = shard_idx * DATA_SHARD_SIZE

                # 对当前分片计算距离
                dot_products = torch.matmul(query_batch, shard_data.t())
                dists_sq = torch.clamp(query_norms + shard_norms_t - 2 * dot_products, min=0.0)
                topk_dists_sq, topk_indices_local = torch.topk(dists_sq, k=TOP_K, dim=1, largest=False)
                
                # 转换本地索引为全局索引
                topk_indices_global = topk_indices_local + shard_start_global
                topk_dists = torch.sqrt(topk_dists_sq)

                # 与之前分片的结果合并，维持全局Top-K
                if batch_topk_indices is None:
                    batch_topk_indices = topk_indices_global
                    batch_topk_dists = topk_dists
                else:
                    # 拼接本分片与之前分片的结果
                    combined_indices = torch.cat([batch_topk_indices, topk_indices_global], dim=1)
                    combined_dists = torch.cat([batch_topk_dists, topk_dists], dim=1)
                    
                    # 对拼接后的 2*TOP_K 个结果再次TopK筛选
                    topk_dists_final, topk_order = torch.topk(combined_dists, k=TOP_K, dim=1, largest=False)
                    
                    # 使用 gather 按排序结果重新获取索引
                    batch_topk_indices = torch.gather(combined_indices, dim=1, index=topk_order)
                    batch_topk_dists = topk_dists_final

            final_indices_tensors.append(batch_topk_indices)
            final_dists_tensors.append(batch_topk_dists)

    # 5. 循环结束后：执行一次大规模的同步和转换
    print("\n搜索完成，正在进行最终数据同步与整理...")
    
    final_indices_gpu = torch.cat(final_indices_tensors, dim=0)
    final_dists_gpu = torch.cat(final_dists_tensors, dim=0)

    final_indices_cpu = final_indices_gpu.cpu().tolist()
    final_dists_cpu = final_dists_gpu.cpu().tolist()
    
    all_results = []
    for i in tqdm(range(total_vectors), desc="Formatting JSON"):
        all_results.append({
            "id": i,
            "top_k_ids": final_indices_cpu[i],
            "top_k_dists": final_dists_cpu[i]
        })

    # 6. 保存结果
    print(f"正在保存到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=None)

    print("完成！")

if __name__ == "__main__":
    main()