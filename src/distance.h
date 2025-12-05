#ifndef DISTANCE_H
#define DISTANCE_H

#ifdef __AVX2__
#include <immintrin.h>
#endif

// 100维专用 - 完全展开，无任何分支

#if defined(__AVX2__)

// 100 = 12*8 + 4
static inline float l2sq_100d(const float* __restrict a, const float* __restrict b) {
    if (ENABLE_RUNTIME_DIST_COUNTING.load(std::memory_order_relaxed)) ++tl_dist_counter;
    
    // 前96维：12个AVX2向量，两路并行
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a+8), _mm256_loadu_ps(b+8));
    __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a+16), _mm256_loadu_ps(b+16));
    __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a+24), _mm256_loadu_ps(b+24));
    __m256 d4 = _mm256_sub_ps(_mm256_loadu_ps(a+32), _mm256_loadu_ps(b+32));
    __m256 d5 = _mm256_sub_ps(_mm256_loadu_ps(a+40), _mm256_loadu_ps(b+40));
    __m256 d6 = _mm256_sub_ps(_mm256_loadu_ps(a+48), _mm256_loadu_ps(b+48));
    __m256 d7 = _mm256_sub_ps(_mm256_loadu_ps(a+56), _mm256_loadu_ps(b+56));
    __m256 d8 = _mm256_sub_ps(_mm256_loadu_ps(a+64), _mm256_loadu_ps(b+64));
    __m256 d9 = _mm256_sub_ps(_mm256_loadu_ps(a+72), _mm256_loadu_ps(b+72));
    __m256 d10 = _mm256_sub_ps(_mm256_loadu_ps(a+80), _mm256_loadu_ps(b+80));
    __m256 d11 = _mm256_sub_ps(_mm256_loadu_ps(a+88), _mm256_loadu_ps(b+88));
    
    // FMA两路并行减少依赖链
    __m256 s0 = _mm256_mul_ps(d0, d0);
    __m256 s1 = _mm256_mul_ps(d1, d1);
    s0 = _mm256_fmadd_ps(d2, d2, s0);
    s1 = _mm256_fmadd_ps(d3, d3, s1);
    s0 = _mm256_fmadd_ps(d4, d4, s0);
    s1 = _mm256_fmadd_ps(d5, d5, s1);
    s0 = _mm256_fmadd_ps(d6, d6, s0);
    s1 = _mm256_fmadd_ps(d7, d7, s1);
    s0 = _mm256_fmadd_ps(d8, d8, s0);
    s1 = _mm256_fmadd_ps(d9, d9, s1);
    s0 = _mm256_fmadd_ps(d10, d10, s0);
    s1 = _mm256_fmadd_ps(d11, d11, s1);
    
    __m256 sum = _mm256_add_ps(s0, s1);
    
    // 水平求和
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 r = _mm_add_ps(lo, hi);
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float s = _mm_cvtss_f32(r);
    
    // 尾部4个用SSE
    __m128 ta = _mm_loadu_ps(a + 96);
    __m128 tb = _mm_loadu_ps(b + 96);
    __m128 td = _mm_sub_ps(ta, tb);
    td = _mm_mul_ps(td, td);
    td = _mm_hadd_ps(td, td);
    td = _mm_hadd_ps(td, td);
    
    return s + _mm_cvtss_f32(td);
}

// ---------------------------------------------------------
// 2-way Batching: 同时计算 (Query <-> A) 和 (Query <-> B) 的距离
// 正收益原理：当加载 A 发生 Cache Miss 时，CPU 可以继续发射加载 B 的指令，
// 且计算指令 FMA 可以交错执行，打破依赖链。
// ---------------------------------------------------------
static inline void l2sq_100d_2x(const float* __restrict q, 
                                const float* __restrict a, 
                                const float* __restrict b,
                                float& out_a, float& out_b) {
    if (ENABLE_RUNTIME_DIST_COUNTING.load(std::memory_order_relaxed)) tl_dist_counter += 2;

    // 累加器
    __m256 sum_a = _mm256_setzero_ps();
    __m256 sum_b = _mm256_setzero_ps();

    // 前96维：完全展开，交错加载和计算以打破依赖链
    // 每次循环处理16维（2个8维块），交错处理 A 和 B
    
    // Block 0-15
    __m256 vq0 = _mm256_loadu_ps(q);
    __m256 vq1 = _mm256_loadu_ps(q + 8);
    __m256 va0 = _mm256_loadu_ps(a);
    __m256 vb0 = _mm256_loadu_ps(b);
    __m256 da0 = _mm256_sub_ps(vq0, va0);
    __m256 db0 = _mm256_sub_ps(vq0, vb0);
    sum_a = _mm256_fmadd_ps(da0, da0, sum_a);
    sum_b = _mm256_fmadd_ps(db0, db0, sum_b);
    
    __m256 va1 = _mm256_loadu_ps(a + 8);
    __m256 vb1 = _mm256_loadu_ps(b + 8);
    __m256 da1 = _mm256_sub_ps(vq1, va1);
    __m256 db1 = _mm256_sub_ps(vq1, vb1);
    sum_a = _mm256_fmadd_ps(da1, da1, sum_a);
    sum_b = _mm256_fmadd_ps(db1, db1, sum_b);
    
    // Block 16-31
    __m256 vq2 = _mm256_loadu_ps(q + 16);
    __m256 vq3 = _mm256_loadu_ps(q + 24);
    __m256 va2 = _mm256_loadu_ps(a + 16);
    __m256 vb2 = _mm256_loadu_ps(b + 16);
    __m256 da2 = _mm256_sub_ps(vq2, va2);
    __m256 db2 = _mm256_sub_ps(vq2, vb2);
    sum_a = _mm256_fmadd_ps(da2, da2, sum_a);
    sum_b = _mm256_fmadd_ps(db2, db2, sum_b);
    
    __m256 va3 = _mm256_loadu_ps(a + 24);
    __m256 vb3 = _mm256_loadu_ps(b + 24);
    __m256 da3 = _mm256_sub_ps(vq3, va3);
    __m256 db3 = _mm256_sub_ps(vq3, vb3);
    sum_a = _mm256_fmadd_ps(da3, da3, sum_a);
    sum_b = _mm256_fmadd_ps(db3, db3, sum_b);
    
    // Block 32-47
    __m256 vq4 = _mm256_loadu_ps(q + 32);
    __m256 vq5 = _mm256_loadu_ps(q + 40);
    __m256 va4 = _mm256_loadu_ps(a + 32);
    __m256 vb4 = _mm256_loadu_ps(b + 32);
    __m256 da4 = _mm256_sub_ps(vq4, va4);
    __m256 db4 = _mm256_sub_ps(vq4, vb4);
    sum_a = _mm256_fmadd_ps(da4, da4, sum_a);
    sum_b = _mm256_fmadd_ps(db4, db4, sum_b);
    
    __m256 va5 = _mm256_loadu_ps(a + 40);
    __m256 vb5 = _mm256_loadu_ps(b + 40);
    __m256 da5 = _mm256_sub_ps(vq5, va5);
    __m256 db5 = _mm256_sub_ps(vq5, vb5);
    sum_a = _mm256_fmadd_ps(da5, da5, sum_a);
    sum_b = _mm256_fmadd_ps(db5, db5, sum_b);
    
    // Block 48-63
    __m256 vq6 = _mm256_loadu_ps(q + 48);
    __m256 vq7 = _mm256_loadu_ps(q + 56);
    __m256 va6 = _mm256_loadu_ps(a + 48);
    __m256 vb6 = _mm256_loadu_ps(b + 48);
    __m256 da6 = _mm256_sub_ps(vq6, va6);
    __m256 db6 = _mm256_sub_ps(vq6, vb6);
    sum_a = _mm256_fmadd_ps(da6, da6, sum_a);
    sum_b = _mm256_fmadd_ps(db6, db6, sum_b);
    
    __m256 va7 = _mm256_loadu_ps(a + 56);
    __m256 vb7 = _mm256_loadu_ps(b + 56);
    __m256 da7 = _mm256_sub_ps(vq7, va7);
    __m256 db7 = _mm256_sub_ps(vq7, vb7);
    sum_a = _mm256_fmadd_ps(da7, da7, sum_a);
    sum_b = _mm256_fmadd_ps(db7, db7, sum_b);
    
    // Block 64-79
    __m256 vq8 = _mm256_loadu_ps(q + 64);
    __m256 vq9 = _mm256_loadu_ps(q + 72);
    __m256 va8 = _mm256_loadu_ps(a + 64);
    __m256 vb8 = _mm256_loadu_ps(b + 64);
    __m256 da8 = _mm256_sub_ps(vq8, va8);
    __m256 db8 = _mm256_sub_ps(vq8, vb8);
    sum_a = _mm256_fmadd_ps(da8, da8, sum_a);
    sum_b = _mm256_fmadd_ps(db8, db8, sum_b);
    
    __m256 va9 = _mm256_loadu_ps(a + 72);
    __m256 vb9 = _mm256_loadu_ps(b + 72);
    __m256 da9 = _mm256_sub_ps(vq9, va9);
    __m256 db9 = _mm256_sub_ps(vq9, vb9);
    sum_a = _mm256_fmadd_ps(da9, da9, sum_a);
    sum_b = _mm256_fmadd_ps(db9, db9, sum_b);
    
    // Block 80-95
    __m256 vq10 = _mm256_loadu_ps(q + 80);
    __m256 vq11 = _mm256_loadu_ps(q + 88);
    __m256 va10 = _mm256_loadu_ps(a + 80);
    __m256 vb10 = _mm256_loadu_ps(b + 80);
    __m256 da10 = _mm256_sub_ps(vq10, va10);
    __m256 db10 = _mm256_sub_ps(vq10, vb10);
    sum_a = _mm256_fmadd_ps(da10, da10, sum_a);
    sum_b = _mm256_fmadd_ps(db10, db10, sum_b);
    
    __m256 va11 = _mm256_loadu_ps(a + 88);
    __m256 vb11 = _mm256_loadu_ps(b + 88);
    __m256 da11 = _mm256_sub_ps(vq11, va11);
    __m256 db11 = _mm256_sub_ps(vq11, vb11);
    sum_a = _mm256_fmadd_ps(da11, da11, sum_a);
    sum_b = _mm256_fmadd_ps(db11, db11, sum_b);

    // Horizontal Reduction for A
    __m128 lo_a = _mm256_castps256_ps128(sum_a);
    __m128 hi_a = _mm256_extractf128_ps(sum_a, 1);
    __m128 sum128_a = _mm_add_ps(lo_a, hi_a);
    sum128_a = _mm_hadd_ps(sum128_a, sum128_a);
    sum128_a = _mm_hadd_ps(sum128_a, sum128_a);
    out_a = _mm_cvtss_f32(sum128_a);

    // Horizontal Reduction for B
    __m128 lo_b = _mm256_castps256_ps128(sum_b);
    __m128 hi_b = _mm256_extractf128_ps(sum_b, 1);
    __m128 sum128_b = _mm_add_ps(lo_b, hi_b);
    sum128_b = _mm_hadd_ps(sum128_b, sum128_b);
    sum128_b = _mm_hadd_ps(sum128_b, sum128_b);
    out_b = _mm_cvtss_f32(sum128_b);

    // 处理尾部 4 维 (96-99)
    __m128 tq = _mm_loadu_ps(q + 96);
    __m128 ta = _mm_loadu_ps(a + 96);
    __m128 tb = _mm_loadu_ps(b + 96);
    __m128 tda = _mm_sub_ps(tq, ta);
    __m128 tdb = _mm_sub_ps(tq, tb);
    tda = _mm_mul_ps(tda, tda);
    tdb = _mm_mul_ps(tdb, tdb);
    tda = _mm_hadd_ps(tda, tda);
    tda = _mm_hadd_ps(tda, tda);
    tdb = _mm_hadd_ps(tdb, tdb);
    tdb = _mm_hadd_ps(tdb, tdb);
    
    out_a += _mm_cvtss_f32(tda);
    out_b += _mm_cvtss_f32(tdb);
}

#else

static inline float l2sq_100d(const float* __restrict a, const float* __restrict b) {
    if (ENABLE_RUNTIME_DIST_COUNTING.load(std::memory_order_relaxed)) ++tl_dist_counter;
    float s = 0.0f;
    for (int i = 0; i < 100; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

static inline void l2sq_100d_2x(const float* __restrict q, 
                                const float* __restrict a, 
                                const float* __restrict b,
                                float& out_a, float& out_b) {
    if (ENABLE_RUNTIME_DIST_COUNTING.load(std::memory_order_relaxed)) tl_dist_counter += 2;
    out_a = 0.0f;
    out_b = 0.0f;
    for (int i = 0; i < 100; ++i) {
        float da = q[i] - a[i]; out_a += da * da;
        float db = q[i] - b[i]; out_b += db * db;
    }
}

#endif

// 主入口 - 直接调用100维版本
static inline float l2sq_dispatch(const float* __restrict a, const float* __restrict b, int /*dim*/) {
    return l2sq_100d(a, b);
}

// 兼容旧接口
static inline float l2sq_simd(const float* __restrict a, const float* __restrict b, int dim) {
    return l2sq_dispatch(a, b, dim);
}

static inline float l2sq_scalar(const float* __restrict a, const float* __restrict b, int /*dim*/) {
    if (ENABLE_RUNTIME_DIST_COUNTING.load(std::memory_order_relaxed)) ++tl_dist_counter;
    float s = 0.0f;
    for (int i = 0; i < 100; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

// VisitedList 和 TagVisitedList 已在 MySolution.h 中定义，这里不再重复

#endif // DISTANCE_H
