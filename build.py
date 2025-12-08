import numpy as np
import os
import struct
import math
import sys
import hashlib
from typing import Tuple, List
try:
    import torch
except ImportError:
    torch = None

# --- HNSW 配置和缓存文件元数据 (严格遵循 C++ cache.h) ---
DIM: int = 100 
M_UPPER: int = 64  # C++ 中的 M 参数 (max_m_upper)
MAX_M_L0: int = M_UPPER * 2  # C++ 中的 max_m
INDEX_VERSION: int = 5 
MAGIC_NUMBER: int = 0x48535746  # 'HSWF'
EFC_CONSTRUCTION: int = 2000  # C++ build.h 中用于缓存路径的 efc 参数，默认设为 100

# --- C++ 缓存路径生成函数模拟 ---

def get_index_cache_path_py(n: int, d: int, M: int, max_layer: int, efc: int) -> str:
    """
    模拟 C++ 中的 get_index_cache_path 函数生成缓存文件路径。
    格式: cache/hnsw_n<N>_d<DIM>_M<M_UPPER>_L<MAX_LEVEL>_efc<EFC>.idx
    """
    filename = f"hnsw_n{n}_d{d}_M{M}_L{max_layer}_efc{efc}.idx"
    return os.path.join("cache", filename)

# --- 2. 数据加载函数 ---

def load_data_from_txt(file_path: str, D: int) -> Tuple[np.ndarray, int]:
    """从TXT文件中加载向量数据，返回 (N x D) 的 float32 数组和节点数 N"""
    print(f"Loading data from {file_path}...")
    try:
        base_vectors = np.genfromtxt(file_path, dtype=np.float32)
    except FileNotFoundError:
        print(f"❌ 错误: 输入文件未找到于 {file_path}. 请检查路径.")
        return np.array([]), 0
    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return np.array([]), 0

    if base_vectors.ndim == 1 and base_vectors.size > 0:
        base_vectors = base_vectors.reshape(1, -1)
        
    N = base_vectors.shape[0]
    if N == 0 or base_vectors.shape[1] != D:
         print(f"❌ 错误: 维度不匹配. 已加载 {N} 个向量, 期望维度 {D}, 实际维度 {base_vectors.shape[1]}.")
         return np.array([]), 0
    
    print(f"✅ 已加载 {N} 个维度为 {D} 的向量.")
    return base_vectors, N

# --- 3. HNSW 结构体模拟与缓存文件生成 ---

def _hash_signature(vec: np.ndarray) -> Tuple[np.uint64, np.uint64]:
    digest = hashlib.sha256(vec.tobytes()).digest()
    sig0, sig1 = struct.unpack_from('<QQ', digest)
    return np.uint64(sig0), np.uint64(sig1)

def build_and_write_hnsw_cache(base_vectors: np.ndarray, N: int):
	print("开始 CUDA K-NN 模拟 L0 链接...")
	np.random.seed(0)
	progress_pct = -1
	def _report_progress(processed: int):
		nonlocal progress_pct
		if N == 0:
			return
		pct = min(100, (processed * 100) // N)
		if pct - progress_pct >= 5 or processed == N:
			progress_pct = pct
			print(f"[Progress] {pct:.0f}% ({processed}/{N}) nodes processed")
	
	log_M = math.log(M_UPPER)
	node_levels = np.zeros(N, dtype=np.int32)
	for i in range(N):
		node_levels[i] = int(-math.log(np.random.rand()) / log_M) if N > 0 else 0
	
	MAX_LEVEL = np.max(node_levels).item() if N > 0 else 0
	ENTER_POINT = np.argmax(node_levels).item() if N > 0 else -1
	
	neighbors_per_node: List[List[int]] = [[] for _ in range(N)]
	torch_vectors = None
	use_torch_cuda = False
	if torch is not None and torch.cuda.is_available():
		try:
			torch_vectors = torch.from_numpy(base_vectors).cuda(non_blocking=True)
			use_torch_cuda = True
			print("✅ PyTorch CUDA 距离计算就绪.")
		except RuntimeError as exc:
			print(f"⚠️ PyTorch CUDA 初始化失败: {exc}. 退回 CPU 分块计算.")
			torch_vectors = None

	def _choose_chunk_size(points: int) -> int:
		if points > 500_000:
			return 64
		if points > 200_000:
			return 128
		if points > 50_000:
			return 256
		return min(512, points or 1)

	chunk_size = _choose_chunk_size(N)
	k = min(N, MAX_M_L0 + 1)
	current_offset = 0

	if use_torch_cuda and torch_vectors is not None:
		total_vectors = torch_vectors
		for chunk_start in range(0, N, chunk_size):
			chunk_end = min(N, chunk_start + chunk_size)
			query_block = total_vectors[chunk_start:chunk_end]
			dists = torch.cdist(query_block, total_vectors, p=2)
			topk_count = min(dists.shape[1], k)
			indices = torch.topk(dists, topk_count, largest=False, sorted=True).indices
			for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
				row = indices[local_idx].tolist()
				node_neighbors = []
				for candidate in row:
					if candidate == global_idx:
						continue
					node_neighbors.append(int(candidate))
					if len(node_neighbors) == MAX_M_L0:
						break
				neighbors_per_node[global_idx] = node_neighbors
			_report_progress(min(chunk_end, N))
	else:
		for global_idx in range(N):
			q = base_vectors[global_idx]
			diff = base_vectors - q
			dists = np.einsum('ij,ij->i', diff, diff)
			topk = min(k, dists.shape[0])
			if topk <= 1:
				neighbors_per_node[global_idx] = []
				_report_progress(global_idx + 1)
				continue
			candidates = np.argpartition(dists, topk - 1)[:topk]
			sorted_candidates = candidates[np.argsort(dists[candidates])]
			node_neighbors = []
			for candidate in sorted_candidates:
				if candidate == global_idx:
					continue
				node_neighbors.append(int(candidate))
				if len(node_neighbors) == MAX_M_L0:
					break
			neighbors_per_node[global_idx] = node_neighbors
			_report_progress(global_idx + 1)
	_report_progress(N)

	l0_offsets = np.zeros(N + 1, dtype=np.uint64)
	l0_links_list: List[int] = []
	for i in range(N):
		l0_offsets[i] = current_offset
		neighbors = neighbors_per_node[i]
		l0_links_list.extend(neighbors)
		current_offset += len(neighbors)
	l0_offsets[N] = current_offset
	l0_links = np.array(l0_links_list, dtype=np.int32) 

	label_lookup = np.arange(N, dtype=np.int32)
	upper_link_offsets = np.full(N * (MAX_LEVEL + 1), -1, dtype=np.int32)
	upper_storage_raw: List[int] = []
	for node in range(N):
		neighbors = neighbors_per_node[node]
		for level in range(1, MAX_LEVEL + 1):
			offset_idx = node * (MAX_LEVEL + 1) + level
			if level > node_levels[node]:
				continue
			filtered = [nbr for nbr in neighbors if node_levels[nbr] >= level][:M_UPPER]
			offset = len(upper_storage_raw)
			upper_link_offsets[offset_idx] = offset
			upper_storage_raw.append(len(filtered))
			upper_storage_raw.extend(filtered)
	upper_link_storage = np.array(upper_storage_raw, dtype=np.int32)

	signatures = np.zeros(N * 2, dtype=np.uint64)
	for idx in range(N):
		sig0, sig1 = _hash_signature(base_vectors[idx])
		signatures[2 * idx] = sig0
		signatures[2 * idx + 1] = sig1

	# --- 3.3 写入缓存文件 (严格遵循 C++ save_flat_index 格式) ---
	
	cache_file = get_index_cache_path_py(
		N, DIM, M_UPPER, MAX_LEVEL, EFC_CONSTRUCTION
	)
	print(f"正在写入缓存文件到 {cache_file}...")
	# 创建 'cache' 目录
	os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
	with open(cache_file, 'wb') as f:
		f.write(struct.pack('I', MAGIC_NUMBER))
		f.write(struct.pack('I', INDEX_VERSION))
		f.write(struct.pack('i', DIM))
		f.write(struct.pack('i', MAX_M_L0))
		f.write(struct.pack('i', M_UPPER))
		f.write(struct.pack('i', ENTER_POINT))
		f.write(struct.pack('i', N))
		f.write(struct.pack('i', MAX_LEVEL))
		f.write(struct.pack('Q', base_vectors.size))
		f.write(base_vectors.tobytes())
		f.write(struct.pack('Q', l0_offsets.size))
		f.write(l0_offsets.tobytes())
		f.write(struct.pack('Q', l0_links.size))
		f.write(l0_links.tobytes())
		f.write(struct.pack('Q', node_levels.size))
		f.write(node_levels.tobytes())
		f.write(struct.pack('Q', upper_link_offsets.size))
		f.write(upper_link_offsets.tobytes())
		f.write(struct.pack('Q', upper_link_storage.size))
		f.write(upper_link_storage.tobytes())
		f.write(struct.pack('Q', label_lookup.size))
		f.write(label_lookup.tobytes())
		f.write(struct.pack('Q', signatures.size))
		f.write(signatures.tobytes())
	print(f"🎉 缓存文件生成成功.")
	print(f"文件路径: {os.path.abspath(cache_file)}")
	print(f"文件大小: {os.path.getsize(cache_file) / (1024*1024):.2f} MB")
	return cache_file, MAX_LEVEL


# ---------------------------------------------------------------------

if __name__ == "__main__":
    cuda_available = torch is not None and torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    # --- 1. 预设文件路径 ---
    INPUT_FILE_PATH = 'data_o/glove/base.txt'
    
    # 创建模拟输入文件 (如果不存在，方便测试)
    os.makedirs(os.path.dirname(INPUT_FILE_PATH) or '.', exist_ok=True)
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"⚠️ {INPUT_FILE_PATH} 未找到. 正在创建一个虚拟文件 (1000个向量, D={DIM}) 用于演示.")
        DUMMY_VECTORS = 1000
        dummy_data = np.random.rand(DUMMY_VECTORS, DIM).astype(np.float32)
        try:
            with open(INPUT_FILE_PATH, 'w') as f:
                for row in dummy_data:
                    f.write(" ".join([f"{x:.6e}" for x in row]) + "\n")
            print("虚拟文件生成完成.")
        except IOError as e:
            print(f"致命错误：无法写入虚拟文件: {e}")
            sys.exit(1)

    # --- 2. 加载数据 ---
    base_vectors, N = load_data_from_txt(INPUT_FILE_PATH, DIM)
    if N == 0:
        sys.exit(1)

    cache_file, max_level = build_and_write_hnsw_cache(base_vectors, N)
    print(f"最终缓存保存在: {cache_file}, 使用 max_level={max_level}")