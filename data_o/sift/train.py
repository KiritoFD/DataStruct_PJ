import numpy as np
import pickle
import struct
from sklearn.cluster import MiniBatchKMeans
import time
import os

class ProductQuantizer:
    def __init__(self, M=8, Ks=256):
        self.M = M
        self.Ks = Ks
        self.sub_len = None
        self.codebooks = []
        self.trained = False
        
    def train(self, data, iterations=100, batch_size=10000):
        n, d = data.shape
        assert d % self.M == 0, "维度必须能被M整除"
        
        self.sub_len = d // self.M
        self.codebooks = []
        
        print(f"训练PQ量化器: M={self.M}, Ks={self.Ks}, 子空间维度={self.sub_len}")
        
        for m in range(self.M):
            start_idx = m * self.sub_len
            end_idx = (m + 1) * self.sub_len
            sub_data = data[:, start_idx:end_idx]
            
            print(f"训练子空间 {m+1}/{self.M}...")
            kmeans = MiniBatchKMeans(n_clusters=self.Ks, batch_size=batch_size,
                                   max_iter=iterations, random_state=42, n_init=3)
            kmeans.fit(sub_data)
            
            self.codebooks.append(kmeans.cluster_centers_)
        
        self.trained = True
        print("PQ训练完成")
    
    def encode_batch(self, data, batch_size=50000):
        """分批编码数据，避免内存爆炸"""
        if not self.trained:
            raise ValueError("PQ量化器尚未训练")
        
        n, d = data.shape
        codes = np.zeros((n, self.M), dtype=np.uint8)
        
        for m in range(self.M):
            print(f"编码子空间 {m+1}/{self.M}...")
            start_idx = m * self.sub_len
            end_idx = (m + 1) * self.sub_len
            sub_data = data[:, start_idx:end_idx]
            codebook = self.codebooks[m]  # [Ks, sub_len]
            
            # 分批处理
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_data = sub_data[start:end]  # [batch_size, sub_len]
                
                # 计算批次内每个点到所有聚类中心的距离
                # 使用向量化但内存友好的方式
                distances = np.zeros((batch_data.shape[0], self.Ks))
                
                for k in range(self.Ks):
                    # 计算到第k个聚类中心的距离
                    diff = batch_data - codebook[k]  # 广播到 [batch_size, sub_len]
                    distances[:, k] = np.sum(diff ** 2, axis=1)  # 平方距离
                
                # 找到最近的聚类中心
                codes[start:end, m] = np.argmin(distances, axis=1)
                
                if (start // batch_size) % 10 == 0:
                    print(f"  进度: {end}/{n} ({end/n*100:.1f}%)")
        
        return codes
    
    def encode_optimized(self, data, batch_size=50000):
        """更优化的编码版本，进一步减少内存使用"""
        if not self.trained:
            raise ValueError("PQ量化器尚未训练")
        
        n, d = data.shape
        codes = np.zeros((n, self.M), dtype=np.uint8)
        
        for m in range(self.M):
            print(f"编码子空间 {m+1}/{self.M}...")
            start_idx = m * self.sub_len
            end_idx = (m + 1) * self.sub_len
            sub_data = data[:, start_idx:end_idx]
            codebook = self.codebooks[m]
            
            # 分批处理
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_data = sub_data[start:end]
                
                # 一次计算一个点的最近邻，避免大矩阵
                for i in range(batch_data.shape[0]):
                    point = batch_data[i]
                    # 计算到所有聚类中心的距离
                    min_dist = float('inf')
                    best_code = 0
                    
                    for k in range(self.Ks):
                        dist = np.sum((point - codebook[k]) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            best_code = k
                    
                    codes[start + i, m] = best_code
                
                if (start // batch_size) % 20 == 0:
                    print(f"  进度: {end}/{n} ({end/n*100:.1f}%)")
        
        return codes
    
    def encode(self, data, batch_size=50000, method='batch'):
        """编码入口函数"""
        if method == 'batch':
            return self.encode_batch(data, batch_size)
        else:
            return self.encode_optimized(data, batch_size)
    
    def decode(self, codes):
        n = codes.shape[0]
        decoded = np.zeros((n, self.M * self.sub_len), dtype=np.float32)
        
        for m in range(self.M):
            for i in range(n):
                decoded[i, m*self.sub_len:(m+1)*self.sub_len] = \
                    self.codebooks[m][codes[i, m]]
        
        return decoded

def load_txt_data(file_path, max_samples=None):
    print(f"从 {file_path} 加载数据...")
    data = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
                
            numbers = line.strip().split()
            if len(numbers) == 128:
                vector = [float(x) for x in numbers]
                data.append(vector)
            
            if (i + 1) % 100000 == 0:
                print(f"已加载 {i + 1} 行数据...")
    
    data = np.array(data, dtype=np.float32)
    print(f"数据加载完成: {data.shape}")
    return data

def save_pq_for_cpp(pq, output_dir="pq_data"):
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        'M': pq.M,
        'Ks': pq.Ks,
        'sub_len': pq.sub_len,
        'original_dim': pq.M * pq.sub_len
    }
    
    with open(f"{output_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    with open(f"{output_dir}/codebooks.bin", 'wb') as f:
        f.write(struct.pack('I', len(pq.codebooks)))
        
        for i, codebook in enumerate(pq.codebooks):
            f.write(struct.pack('II', codebook.shape[0], codebook.shape[1]))
            codebook.astype(np.float32).tofile(f)
    
    with open(f"{output_dir}/metadata.txt", 'w') as f:
        f.write(f"M {pq.M}\n")
        f.write(f"Ks {pq.Ks}\n")
        f.write(f"sub_len {pq.sub_len}\n")
        f.write(f"original_dim {pq.M * pq.sub_len}\n")
    
    print(f"PQ数据已保存到 {output_dir}")

def save_encoded_data(codes, output_dir="pq_data"):
    with open(f"{output_dir}/encoded_codes.bin", 'wb') as f:
        f.write(struct.pack('II', codes.shape[0], codes.shape[1]))
        codes.tofile(f)
    
    print(f"编码数据已保存: {codes.shape}")

def main():
    # 参数配置
    M = 8
    Ks = 256
    
    # 1. 加载TXT数据
    txt_file_path = "base.txt"
    print("加载TXT数据...")
    
    data = load_txt_data(txt_file_path)
    n_samples, dim = data.shape
    
    print(f"数据统计: {n_samples} 个样本, {dim} 维")
    
    # 2. 训练PQ量化器
    print("开始训练PQ...")
    pq = ProductQuantizer(M=M, Ks=Ks)
    
    start_time = time.time()
    pq.train(data, iterations=100, batch_size=10000)
    training_time = time.time() - start_time
    print(f"PQ训练耗时: {training_time:.2f} 秒")
    
    # 3. 分批编码所有数据（使用优化版本）
    print("编码数据（使用分批处理避免内存不足）...")
    start_time = time.time()
    
    # 尝试使用内存友好的编码方法
    try:
        codes = pq.encode(data, batch_size=20000, method='optimized')
    except MemoryError:
        print("内存不足，使用更保守的批次大小...")
        codes = pq.encode(data, batch_size=10000, method='optimized')
    
    encoding_time = time.time() - start_time
    print(f"数据编码耗时: {encoding_time:.2f} 秒")
    
    # 4. 计算重构误差（使用样本计算）
    print("计算重构误差（使用样本）...")
    sample_size = min(10000, n_samples)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    sample_data = data[sample_indices]
    sample_codes = codes[sample_indices]
    
    reconstructed = pq.decode(sample_codes)
    error = np.mean(np.linalg.norm(sample_data - reconstructed, axis=1))
    print(f"样本平均重构误差: {error:.6f}")
    
    # 5. 保存结果
    output_dir = "pq_128d_data"
    save_pq_for_cpp(pq, output_dir)
    save_encoded_data(codes, output_dir)
    
    # 6. 打印压缩信息
    original_size = n_samples * dim * 4
    compressed_size = n_samples * M * 1
    compression_ratio = original_size / compressed_size
    
    print(f"\n=== 压缩统计 ===")
    print(f"原始数据大小: {original_size / (1024*1024):.2f} MB")
    print(f"压缩后大小: {compressed_size / (1024*1024):.2f} MB")
    print(f"压缩比: {compression_ratio:.2f}x")
    
    # 保存Python对象
    with open(f"{output_dir}/pq_quantizer.pkl", 'wb') as f:
        pickle.dump(pq, f)
    
    print(f"\n所有文件已保存到目录: {output_dir}")

if __name__ == "__main__":
    main()