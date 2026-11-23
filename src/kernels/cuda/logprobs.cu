// src/kernels/cuda/logprobs.cu - a pick's log probability and its row's likeliest tokens (see the header).
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/pdl.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace basalt::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kPer = kLogprobChunk / kThreads;   // 8 logits per thread, held in registers
constexpr int kPart = 4 + 2 * kLogprobMaxTop;    // a chunk's scratch: max, (pad), exp sum (double), top ids, top values
constexpr int kMaxVocab = 1 << 20;               // the widest head the merge's shared memory holds
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kSentinelId = 0x7fffffff;

struct Cand {
    float v;
    int id;
};

// branch-free selection order: larger value first, ties to the lower id
__device__ __forceinline__ bool cand_first(const Cand& a, const Cand& b) {
    return (a.v > b.v) | ((a.v == b.v) & (a.id < b.id));
}

__device__ __forceinline__ Cand cand_shfl_xor(const Cand& c, int m) {
    return {__shfl_xor_sync(kFullMask, c.v, m), __shfl_xor_sync(kFullMask, c.id, m)};
}

__device__ __forceinline__ Cand cand_exchange(const Cand& c, int stride, bool keep_first) {
    const Cand o = cand_shfl_xor(c, stride);
    return (cand_first(o, c) == keep_first) ? o : c;
}

// 32 pairs (one per lane) sorted in selection order: lane i holds the i-th best
__device__ __forceinline__ Cand warp_sort32(Cand c) {
    const int lane = (int) (threadIdx.x & 31);
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const bool lower = (lane & stride) == 0;
            const bool up = size == 32 || (lane & size) == 0;
            c = cand_exchange(c, stride, lower == up);
        }
    }
    return c;
}

// 32 values sorted descending (largest first)
__device__ __forceinline__ float warp_sort32_values(float v) {
    const int lane = (int) (threadIdx.x & 31);
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const float o = __shfl_xor_sync(kFullMask, v, stride);
            const bool keep_max = ((lane & stride) == 0) == (size == 32 || (lane & size) == 0);
            v = keep_max ? fmaxf(v, o) : fminf(v, o);
        }
    }
    return v;
}

// top 32 of two already-sorted 32-element warp lists, sorted
__device__ __forceinline__ Cand warp_merge32(const Cand& a, const Cand& b) {
    const int lane = (int) (threadIdx.x & 31);
    const Cand rb = {__shfl_sync(kFullMask, b.v, 31 - lane), __shfl_sync(kFullMask, b.id, 31 - lane)};
    Cand c = cand_first(rb, a) ? rb : a;
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) c = cand_exchange(c, stride, (lane & stride) == 0);
    return c;
}

// top 32 of `n` (power of two) sorted 32-element warp lists
template <int n>
__device__ __forceinline__ Cand warp_merge_tree(Cand (&c)[n]) {
#pragma unroll
    for (int w = 1; w < n; w <<= 1) {
#pragma unroll
        for (int i = 0; i + w < n; i += 2 * w) c[i] = warp_merge32(c[i], c[i + w]);
    }
    return c[0];
}

// Warp's first `top` (1 <= top <= 32) of its kPer x 32 values in registers, sorted in lane 0..top-1.
// Uses the sorted lane maxima to find a threshold tau bounding the top-th element from below; when at most 32 valid
// entries reach tau, compacts them into `buf` (32 entries) and runs one 32-wide bitonic sort; otherwise sorts and merges
// all kPer sets of 32 (exact even with massive ties or all -inf).
__device__ __forceinline__ Cand warp_first_k(const float (&x)[kPer], float wm, int tid, int i0, int n, int top, Cand* buf) {
    const int lane = tid & 31;
    const float sm = warp_sort32_values(wm);
    auto live_count = [&](float tau) {
        int cnt = 0;
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            cnt += __popc(__ballot_sync(kFullMask, (tid + j * kThreads < n) && (x[j] >= tau)));
        return cnt;
    };
    int lo = 0, hi = top - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (live_count(__shfl_sync(kFullMask, sm, mid)) >= top) hi = mid;
        else lo = mid + 1;
    }
    const float tau = __shfl_sync(kFullMask, sm, lo);
    unsigned bal[kPer];
    int total = 0;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        bal[j] = __ballot_sync(kFullMask, (tid + j * kThreads < n) && (x[j] >= tau));
        total += __popc(bal[j]);
    }
    if (total <= 32) {
        const unsigned below = (1u << lane) - 1u;
        if (lane >= total) buf[lane] = {-INFINITY, kSentinelId};
        int base = 0;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            if ((bal[j] >> lane) & 1u) buf[base + __popc(bal[j] & below)] = {x[j], i0 + tid + j * kThreads};
            base += __popc(bal[j]);
        }
        __syncwarp();
        const Cand r = buf[lane];
        __syncwarp();
        return warp_sort32(r);
    }
    Cand c[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int idx = tid + j * kThreads;
        c[j] = idx < n ? Cand{x[j], i0 + idx} : Cand{-INFINITY, kSentinelId};
        c[j] = warp_sort32(c[j]);
    }
    return warp_merge_tree<kPer>(c);
}

// chunk `chunk` of row `row`: its max, the sum of exp(x - max) and its `top` likeliest entries (raw logits)
__device__ void chunk_stage(const float* __restrict__ logits, int n_vocab, int top, float* __restrict__ scratch, int chunk,
                            int row, int tid) {
    __shared__ float s_v[kWarps];
    __shared__ double s_d[kWarps];
    __shared__ Cand s_w[kWarps][32];
    const int warp = tid >> 5, lane = tid & 31;
    const int i0 = chunk * kLogprobChunk;
    const int n = min(kLogprobChunk, n_vocab - i0);
    const float* src = logits + (size_t) row * (size_t) n_vocab + i0;

    float x[kPer];
    float mx = -FLT_MAX;
    float wm = -INFINITY;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int idx = tid + j * kThreads;
        if (idx < n) {
            const float v = src[idx];
            x[j] = v;
            mx = fmaxf(mx, v);
            wm = fmaxf(wm, v);
        } else {
            x[j] = -INFINITY;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, o));
    if (lane == 0) s_v[warp] = mx;
    __syncthreads();
    mx = s_v[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) mx = fmaxf(mx, s_v[w]);

    double sum = 0.0;
#pragma unroll
    for (int j = 0; j < kPer; ++j)
        if (tid + j * kThreads < n) sum += (double) expf(x[j] - mx);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(kFullMask, sum, o);
    if (lane == 0) s_d[warp] = sum;

    if (top > 0) s_w[warp][lane] = warp_first_k(x, wm, tid, i0, n, top, s_w[warp]);
    __syncthreads();

    float* part = scratch + ((size_t) row * row_logprobs_chunks(n_vocab) + chunk) * kPart;
    if (top == 0) {
        if (tid == 0) {
            double total = 0.0;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) total += s_d[w];
            part[0] = mx;
            reinterpret_cast<double*>(part)[1] = total;
        }
        return;
    }
    if (warp == 1 && lane == 0) {
        double total = 0.0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) total += s_d[w];
        part[0] = mx;
        reinterpret_cast<double*>(part)[1] = total;
    }
    if (warp == 0) {
        Cand w[kWarps];
#pragma unroll
        for (int k = 0; k < kWarps; ++k) w[k] = s_w[k][lane];
        const Cand res = warp_merge_tree<kWarps>(w);
        if (lane < top) {
            reinterpret_cast<int*>(part)[4 + lane] = res.id;
            part[4 + kLogprobMaxTop + lane] = res.v;
        }
    }
}

// row `row`: the chunks merged in order, then the top entries merged directly from each chunk's sorted candidate list
__device__ void merge_stage(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ picks, int top,
                            float* __restrict__ out, const float* __restrict__ scratch, int row, int tid) {
    __shared__ float s_v[kWarps];
    __shared__ double s_lse;
    __shared__ double s_c[(kMaxVocab + kLogprobChunk - 1) / kLogprobChunk];
    __shared__ Cand s_w[kWarps][32];
    const int warp = tid >> 5, lane = tid & 31;
    const int nc = row_logprobs_chunks(n_vocab);
    const float* parts = scratch + (size_t) row * nc * kPart;
    float* o = out + (size_t) row * kLogprobStride;

    float mx = -FLT_MAX;
    for (int c = tid; c < nc; c += kThreads) mx = fmaxf(mx, parts[(size_t) c * kPart]);
#pragma unroll
    for (int step = 16; step > 0; step >>= 1) mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, step));
    if (lane == 0) s_v[warp] = mx;
    __syncthreads();
    mx = s_v[0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) mx = fmaxf(mx, s_v[w]);

    for (int c = tid; c < nc; c += kThreads)   // each chunk's sum rescaled to the row's max, one thread a chunk
        s_c[c] = reinterpret_cast<const double*>(parts + (size_t) c * kPart)[1] * exp((double) parts[(size_t) c * kPart] - mx);

    if (top > 0) {
        // each chunk's `top` entries are already sorted: stream them across warps with warp_merge32
        Cand acc{-INFINITY, kSentinelId};
        bool first = true;
        for (int c = warp; c < nc; c += kWarps) {
            const float* part = parts + (size_t) c * kPart;
            Cand cur{-INFINITY, kSentinelId};
            if (lane < top) {
                const int id = reinterpret_cast<const int*>(part)[4 + lane];
                if (id < n_vocab) cur = {part[4 + kLogprobMaxTop + lane], id};
            }
            acc = first ? cur : warp_merge32(acc, cur);
            first = false;
        }
        s_w[warp][lane] = acc;
    }
    __syncthreads();

    if (top == 0) {
        if (tid == 0) {
            double sum = 0.0;
            for (int c = 0; c < nc; ++c) sum += s_c[c];   // in chunk order
            const double lse = (double) mx + log(sum);
            const int pick = picks ? picks[row] : -1;
            o[0] = pick >= 0 && pick < n_vocab ? (float) ((double) logits[(size_t) row * n_vocab + pick] - lse) : -INFINITY;
            o[1] = (float) lse;
        }
        return;
    }

    Cand res{-INFINITY, kSentinelId};
    if (warp == 0) {
        Cand w[kWarps];
#pragma unroll
        for (int k = 0; k < kWarps; ++k) w[k] = s_w[k][lane];
        res = warp_merge_tree<kWarps>(w);
    } else if (warp == 1 && lane == 0) {
        double sum = 0.0;
        for (int c = 0; c < nc; ++c) sum += s_c[c];   // in chunk order
        const double lse = (double) mx + log(sum);
        s_lse = lse;
        const int pick = picks ? picks[row] : -1;
        o[0] = pick >= 0 && pick < n_vocab ? (float) ((double) logits[(size_t) row * n_vocab + pick] - lse) : -INFINITY;
        o[1] = (float) lse;
    }
    __syncthreads();
    if (warp == 0 && lane < top) {
        const double lse = s_lse;
        const bool real = res.id < n_vocab;
        reinterpret_cast<int*>(o)[2 + 2 * lane] = real ? res.id : -1;
        o[3 + 2 * lane] = real ? (float) ((double) res.v - lse) : -INFINITY;
    }
}

__global__ void __launch_bounds__(kThreads) chunk_kernel(const float* logits, int n_vocab, int top, float* scratch) {
    pdl_launch_dependents();
    pdl_wait();
    chunk_stage(logits, n_vocab, top, scratch, (int) blockIdx.x, (int) blockIdx.y, (int) threadIdx.x);
}

__global__ void __launch_bounds__(kThreads) merge_kernel(const float* logits, int n_vocab, const int32_t* picks, int top,
                                                         float* out, const float* scratch) {
    pdl_launch_dependents();
    pdl_wait();
    merge_stage(logits, n_vocab, picks, top, out, scratch, (int) blockIdx.x, (int) threadIdx.x);
}

}  // namespace

void row_logprobs(const float* logits, int n_rows, int n_vocab, const int32_t* picks, int top, float* out, float* scratch,
                  void* stream) {
    if (n_rows <= 0 || n_vocab <= 0 || n_vocab > kMaxVocab) return;
    top = top < 0 ? 0 : top > kLogprobMaxTop ? kLogprobMaxTop : top;
    const cudaStream_t st = (cudaStream_t) stream;
    launch_pdl(chunk_kernel, dim3((unsigned) row_logprobs_chunks(n_vocab), (unsigned) n_rows), dim3(kThreads), 0, st, logits,
               n_vocab, top, scratch);
    launch_pdl(merge_kernel, dim3((unsigned) n_rows), dim3(kThreads), 0, st, logits, n_vocab, picks, top, out, scratch);
}

}  // namespace basalt::kernels
