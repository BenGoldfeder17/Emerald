// tests/kernels/nvfp4_planes_test.cpp - kernels::nvfp4p::from_blocks: every value's E2M1 code and UE4M3 scale of a random
// ggml NVFP4 matrix land where nvfp4_planes.hpp says, and dequantize to the same floats as artifact/dequant.hpp's blocks.
// Exit 77 without a GPU.
#include "basalt/artifact/dequant.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace p = basalt::kernels::nvfp4p;

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    const int64_t rows = 256, K = 640;
    std::mt19937 rng(4);
    std::vector<uint8_t> blk(p::bytes(rows, K));
    for (auto& b : blk) b = (uint8_t) rng();
    for (size_t i = 0; i < blk.size(); i += 36)
        for (int s = 0; s < 4; ++s) blk[i + s] = (uint8_t) (rng() % 0x7F);
    void *d_b, *d_p;
    cudaMalloc(&d_b, blk.size());
    cudaMalloc(&d_p, blk.size());
    cudaMemcpy(d_b, blk.data(), blk.size(), cudaMemcpyHostToDevice);
    p::from_blocks(d_b, rows, K, d_p, nullptr);
    std::vector<uint8_t> pl(blk.size());
    cudaMemcpy(pl.data(), d_p, pl.size(), cudaMemcpyDeviceToHost);
    static const float e2m1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
    int bad = 0;
    std::vector<float> ref(64);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t b = 0; b < K / 64; ++b) {
            basalt::dequantize_nvfp4(&blk[(size_t) (r * (K / 64) + b) * 36], ref.data());
            for (int v = 0; v < 64; ++v) {
                const int64_t k = b * 64 + v;
                const uint8_t byte = pl[(size_t) (r * (K / 2) + k / 2)];
                const int code = (k & 1) ? byte >> 4 : byte & 15;
                const uint8_t sc = pl[p::scale_offset(rows, K) + (size_t) p::scale_index(r, k / 16, K)];
                const float got = e2m1[code] * basalt::ue4m3_to_fp32_half(sc) * 2.0f;
                bad += got != ref[(size_t) v] && !(got == 0.0f && ref[(size_t) v] == 0.0f);
            }
        }
    // and back: the blocks bit for bit
    void* d_b2;
    cudaMalloc(&d_b2, blk.size());
    p::to_blocks(d_p, rows, K, d_b2, nullptr);
    std::vector<uint8_t> back(blk.size());
    cudaMemcpy(back.data(), d_b2, back.size(), cudaMemcpyDeviceToHost);
    const bool round = back == blk;
    // from_f8: both per-row scales (as on an F8 head) and scalar scale reconstruct within NVFP4 quantization error
    std::vector<uint8_t> f8((size_t) rows * K);
    for (auto& b : f8) {
        b = (uint8_t) rng();
        if ((b & 0x7f) == 0x7f) b = 0;   // finite E4M3
    }
    std::vector<float> rsc((size_t) rows);
    for (int64_t r = 0; r < rows; ++r) rsc[(size_t) r] = 0.002f * (1.0f + (float) (r % 7) * 0.3f);
    void *d_f8 = nullptr, *d_fp = nullptr;
    float *d_rsc = nullptr, *d_osc = nullptr;
    cudaMalloc(&d_f8, f8.size());
    cudaMalloc(&d_fp, p::bytes(rows, K));
    cudaMalloc(&d_rsc, rsc.size() * sizeof(float));
    cudaMalloc(&d_osc, sizeof(float));
    cudaMemcpy(d_f8, f8.data(), f8.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rsc, rsc.data(), rsc.size() * sizeof(float), cudaMemcpyHostToDevice);
    p::from_f8(d_f8, nullptr, d_rsc, rows, K, d_fp, d_osc, nullptr);
    std::vector<uint8_t> fpl(p::bytes(rows, K));
    float gs = 0.0f;
    cudaMemcpy(fpl.data(), d_fp, fpl.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&gs, d_osc, sizeof(float), cudaMemcpyDeviceToHost);
    double num = 0.0, den = 0.0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < K; ++k) {
            __nv_fp8_e4m3 fv;
            std::memcpy(&fv, &f8[(size_t) r * K + k], 1);
            const float want = (float) fv * rsc[(size_t) r];
            const uint8_t byte = fpl[(size_t) (r * (K / 2) + k / 2)];
            const int code = (k & 1) ? byte >> 4 : byte & 15;
            const uint8_t sc = fpl[p::scale_offset(rows, K) + (size_t) p::scale_index(r, k / 16, K)];
            const float got = e2m1[code] * basalt::ue4m3_to_fp32_half(sc) * 2.0f * gs;
            num += (double) (got - want) * (got - want);
            den += (double) want * want;
        }
    const double rel_rms = std::sqrt(num / std::max(den, 1e-30));
    const bool f8_ok = rel_rms < 0.15 && gs > 0.0f;
    cudaFree(d_f8); cudaFree(d_fp); cudaFree(d_rsc); cudaFree(d_osc); cudaFree(d_b); cudaFree(d_p); cudaFree(d_b2);
    const bool ok = bad == 0 && round && f8_ok && cudaGetLastError() == cudaSuccess;
    std::printf("nvfp4 planes from ggml blocks: %s (%d values differ); back to blocks: %s; from_f8 rel_rms=%.4f (%s)\n",
                bad == 0 ? "same values" : "WRONG  FAIL", bad, round ? "bitwise" : "DIFFER  FAIL", rel_rms,
                f8_ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
