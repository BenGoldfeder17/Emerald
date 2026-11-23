// tests/kernels/dense_ops_parity.cpp - the dense stack's element-wise kernels (basalt/kernels/dense_ops.hpp) and the block
// drafter's windowed attention (gqa_window_attn) against CPU references in double precision, at the Qwen3.8-27B widths
// (5120 hidden, 17408 FFN) and its DFlash 2 drafter's (32 x 128 query heads, 8 KV heads, a 2048-cell window, groups of
// 16 channels, 16 candidates of rank 256).  Prints max abs / rel error per kernel.  Exit 77 without a GPU.
#include "basalt/kernels/dense_ops.hpp"
#include "basalt/kernels/f16_bits.hpp"
#include "basalt/kernels/gdn_prompt.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/logprobs.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

std::mt19937 rng(11);
std::normal_distribution<float> nd(0.f, 1.f);
cudaStream_t S;
int fails = 0;

template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc(&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaStreamSynchronize(S);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}
std::vector<float> randn(size_t n, float sd = 1.f) {
    std::vector<float> v(n);
    for (auto& x : v) x = sd * nd(rng);
    return v;
}
void report(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    double a = 0, r = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double e = std::fabs(got[i] - want[i]);
        a = std::max(a, e);
        r = std::max(r, e / std::max(std::fabs(want[i]), 1e-3));
    }
    std::printf("%-22s max abs %.3g rel %.3g%s\n", what, a, r, a > tol ? "  FAIL" : "");
    if (a > tol) ++fails;
}
uint16_t bf(float f) {   // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float fb(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) return 77;
    cudaStreamCreate(&S);
    const int H = 5120, FF = 17408, T = 8;
    {   // rms_norm_rows into a strided output
        auto x = randn((size_t) T * H), w = randn(H, 0.2f);
        for (auto& v : w) v += 1.0f;
        float *dx = up(x), *dw = up(w), *dy = nullptr;
        cudaMalloc(&dy, (size_t) T * 2 * H * 4);
        basalt::kernels::rms_norm_rows(dx, dw, dy, T, H, 1e-6f, S, 2 * H);
        auto y = down(dy, (size_t) T * 2 * H);
        std::vector<float> got;
        std::vector<double> want;
        for (int t = 0; t < T; ++t) {
            double ss = 0;
            for (int c = 0; c < H; ++c) ss += (double) x[(size_t) t * H + c] * x[(size_t) t * H + c];
            const double inv = 1.0 / std::sqrt(ss / H + 1e-6);
            for (int c = 0; c < H; ++c) {
                got.push_back(y[(size_t) t * 2 * H + c]);
                want.push_back(x[(size_t) t * H + c] * inv * w[c]);
            }
        }
        report("rms_norm_rows", got, want, 1e-4);
    }
    {   // swiglu_scaled with NVFP4 tensor scales
        auto g = randn((size_t) T * FF, 3.f), u = randn((size_t) T * FF, 3.f);
        std::vector<float> sg = {0.37f}, su = {1.9f};
        float *dg = up(g), *du = up(u), *dsg = up(sg), *dsu = up(su), *dh = nullptr;
        cudaMalloc(&dh, (size_t) T * FF * 4);
        basalt::kernels::swiglu_scaled(dg, du, dsg, dsu, dh, T, FF, FF, S);
        auto h = down(dh, (size_t) T * FF);
        std::vector<double> want(h.size());
        for (size_t i = 0; i < h.size(); ++i) {
            const double a = (double) g[i] * sg[0], b = (double) u[i] * su[0];
            want[i] = a / (1 + std::exp(-a)) * b;
        }
        report("swiglu_scaled", h, want, 1e-4);
    }
    {   // bf16_rows_dot (the GDN gates' and the selector's projections)
        const int N = 96;
        auto x = randn((size_t) T * H);
        std::vector<uint16_t> w((size_t) N * H);
        for (auto& v : w) v = bf(0.02f * nd(rng));
        float* dx = up(x);
        uint16_t* dw = up(w);
        float* dy = nullptr;
        cudaMalloc(&dy, (size_t) T * N * 4);
        basalt::kernels::bf16_rows_dot(dx, dw, dy, T, N, H, S);
        auto y = down(dy, (size_t) T * N);
        std::vector<double> want(y.size());
        for (int t = 0; t < T; ++t)
            for (int n = 0; n < N; ++n) {
                double a = 0;
                for (int k = 0; k < H; ++k) a += (double) x[(size_t) t * H + k] * fb(w[(size_t) n * H + k]);
                want[(size_t) t * N + n] = a;
            }
        report("bf16_rows_dot", y, want, 3e-4);   // tf32 tensor cores: two-term x, but the mma accumulates below fp32
    }
    {   // gdn_gating_proj against bf16_rows_dot + gdn_gates and a double reference, a 2,048-row chunk and a ragged one
        const int HV = 48, N = 2 * HV;
        std::vector<uint16_t> w((size_t) N * H);
        for (auto& v : w) v = bf(0.02f * nd(rng));
        auto dt = randn(HV, 0.5f), sa = randn(HV, 0.3f);
        for (auto& v : sa) v = -std::exp(v);
        uint16_t* dw = up(w);
        float *ddt = up(dt), *dsa = up(sa);
        for (const int Tc : {2048, 37}) {
            auto x = randn((size_t) Tc * H);
            for (size_t i = 0; i < x.size(); i += 97) x[i] *= 8.f;   // a few large activations
            float *dx = up(x), *dg = nullptr, *db = nullptr, *dab = nullptr, *dg0 = nullptr, *db0 = nullptr;
            cudaMalloc(&dg, (size_t) Tc * HV * 4);
            cudaMalloc(&db, (size_t) Tc * HV * 4);
            cudaMalloc(&dg0, (size_t) Tc * HV * 4);
            cudaMalloc(&db0, (size_t) Tc * HV * 4);
            cudaMalloc(&dab, (size_t) Tc * N * 4);
            basalt::kernels::gdn_gating_proj(dx, dw, ddt, dsa, dg, db, Tc, HV, H, S);
            basalt::kernels::bf16_rows_dot(dx, dw, dab, Tc, N, H, S);
            basalt::kernels::gdn_gates(dab, ddt, dsa, dg0, db0, Tc, S);
            auto g = down(dg, (size_t) Tc * HV), b = down(db, (size_t) Tc * HV), g0 = down(dg0, (size_t) Tc * HV),
                 b0 = down(db0, (size_t) Tc * HV);
            std::vector<float> gg, bb, gg0, bb0;
            std::vector<double> wg, wb;
            for (int t = 0; t < Tc; t += (Tc > 100 ? 7 : 1))
                for (int h = 0; h < HV; ++h) {
                    double a = 0, bs = 0;
                    for (int k = 0; k < H; ++k) {
                        a += (double) x[(size_t) t * H + k] * fb(w[(size_t) h * H + k]);
                        bs += (double) x[(size_t) t * H + k] * fb(w[(size_t) (HV + h) * H + k]);
                    }
                    const double v = a + dt[h];
                    wg.push_back((v > 20 ? v : std::log1p(std::exp(v))) * sa[h]);
                    wb.push_back(1.0 / (1.0 + std::exp(-bs)));
                    gg.push_back(g[(size_t) t * HV + h]); bb.push_back(b[(size_t) t * HV + h]);
                    gg0.push_back(g0[(size_t) t * HV + h]); bb0.push_back(b0[(size_t) t * HV + h]);
                }
            std::printf("T = %d  new kernel:\n", Tc);
            report("  gdn_gating_proj g", gg, wg, 3e-4);
            report("  gdn_gating_proj beta", bb, wb, 3e-5);
            std::printf("T = %d  old bf16_rows_dot + gdn_gates:\n", Tc);
            report("  old g", gg0, wg, 3e-4);
            report("  old beta", bb0, wb, 3e-5);
            if (Tc == 2048) {   // microbench
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0); cudaEventCreate(&e1);
                const int R = 200;
                for (int pass = 0; pass < 2; ++pass) {
                    cudaEventRecord(e0, S);
                    for (int r = 0; r < R; ++r) {
                        if (pass == 0) {
                            basalt::kernels::bf16_rows_dot(dx, dw, dab, Tc, N, H, S);
                            basalt::kernels::gdn_gates(dab, ddt, dsa, dg0, db0, Tc, S);
                        } else basalt::kernels::gdn_gating_proj(dx, dw, ddt, dsa, dg, db, Tc, HV, H, S);
                    }
                    cudaEventRecord(e1, S);
                    cudaEventSynchronize(e1);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, e0, e1);
                    std::printf("  bench T=2048 %s: %.1f us per call\n", pass ? "gdn_gating_proj" : "bf16_rows_dot + gdn_gates", 1e3 * ms / R);
                }
            }
        }
    }
    {   // gdn_conv at a 2,048-row chunk: timing and its bandwidth (the fp32 traffic bound: a float4 variant gained 8% and was dropped)
        const int C = 10240, Tc = 2048;
        auto q = randn((size_t) Tc * C), cw = randn((size_t) C * 4, 0.5f), hh = randn((size_t) C * 3);
        float *dq = up(q), *dcw = up(cw), *dh = up(hh), *dout = nullptr;
        cudaMalloc(&dout, (size_t) Tc * C * 4);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, S);
        for (int r = 0; r < 100; ++r) basalt::kernels::gdn_conv(dh, dq, dcw, dout, Tc, S);
        cudaEventRecord(e1, S);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        std::printf("  bench gdn_conv T=2048: %.1f us per call, %.0f GB/s\n", 1e3 * ms / 100, 2.0 * Tc * C * 4 / (ms / 100 * 1e-3) / 1e9);
    }
    {   // dyn_conv2: base + dynamic two-tap convolution along the block, 16-channel groups
        const int G = 16, groups = H / G, LD = 4 * groups;
        auto x = randn((size_t) T * H), dyn = randn((size_t) T * LD, 0.3f), base = randn(2 * (size_t) H, 0.5f);
        float *dx = up(x), *dd = up(dyn), *db = up(base), *dy = nullptr;
        cudaMalloc(&dy, (size_t) T * H * 4);
        basalt::kernels::dyn_conv2(dx, dd + 2 * groups, LD, db, dy, T, H, G, S);   // the second half, as `finish` reads
        auto y = down(dy, (size_t) T * H);
        std::vector<double> want(y.size());
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < H; ++c) {
                const float* dr = &dyn[(size_t) t * LD + 2 * groups];
                double a = ((double) base[c] + dr[c / G]) * x[(size_t) t * H + c];
                if (t > 0) a += ((double) base[H + c] + dr[groups + c / G]) * x[(size_t) (t - 1) * H + c];
                want[(size_t) t * H + c] = a;
            }
        report("dyn_conv2", y, want, 1e-5);
        // fused BF16 store, fused BF16 residual add, and multi-block first_mask (two blocks of 4 rows):
        // compare bitwise against unfused dyn_conv2 (per block) + to_bf16 / add_scaled on GPU
        const uint32_t mask2 = (1u << 0) | (1u << 4);
        float* dy_ref = nullptr;
        uint16_t *dy16 = nullptr, *dy16_ref = nullptr;
        cudaMalloc(&dy_ref, (size_t) T * H * 4);
        cudaMalloc(&dy16, (size_t) T * H * 2);
        cudaMalloc(&dy16_ref, (size_t) T * H * 2);
        basalt::kernels::dyn_conv2(dx, dd + 2 * groups, LD, db, dy_ref, 4, H, G, S);
        basalt::kernels::dyn_conv2(dx + 4 * H, dd + 4 * LD + 2 * groups, LD, db, dy_ref + 4 * H, 4, H, G, S);
        basalt::kernels::to_bf16(dy_ref, dy16_ref, (size_t) T * H, S);
        basalt::kernels::dyn_conv2_bf16(dx, dd + 2 * groups, LD, db, dy16, T, H, G, S, mask2);
        auto y16 = down(dy16, (size_t) T * H);
        auto y16_ref = down(dy16_ref, (size_t) T * H);
        std::vector<uint16_t> res0((size_t) T * H);
        for (auto& v : res0) v = bf(0.5f * nd(rng));
        uint16_t* dres = up(res0);
        uint16_t* dres_ref = up(res0);
        basalt::kernels::add_scaled(dres_ref, dy_ref, nullptr, (size_t) T * H, S);
        basalt::kernels::dyn_conv2_add_bf16(dx, dd + 2 * groups, LD, db, dres, T, H, G, S, mask2);
        auto res1 = down(dres, (size_t) T * H);
        auto res1_ref = down(dres_ref, (size_t) T * H);
        int bf_diff = 0, add_diff = 0;
        for (size_t idx = 0; idx < (size_t) T * H; ++idx) {
            if (y16[idx] != y16_ref[idx]) ++bf_diff;
            if (res1[idx] != res1_ref[idx]) ++add_diff;
        }
        std::printf("dyn_conv2_bf16/add_bf16 bitwise diffs vs unfused: bf16=%d add_bf16=%d%s\n", bf_diff, add_diff,
                    (bf_diff || add_diff) ? "  FAIL" : "");
        if (bf_diff || add_diff) ++fails;
        cudaFree(dy_ref);
        cudaFree(dy16);
        cudaFree(dy16_ref);
        cudaFree(dres);
        cudaFree(dres_ref);
    }
    {   // candidate_path against the same walk on the CPU
        const int V = 4096, R = 256, K = 16, P = 7;
        std::vector<uint16_t> pred((size_t) V * R), succ((size_t) V * R);
        for (auto& v : pred) v = bf(0.3f * nd(rng));
        for (auto& v : succ) v = bf(0.3f * nd(rng));
        auto h = randn((size_t) P * R, 0.3f);
        std::vector<float> lp((size_t) P * basalt::kernels::kLogprobStride, 0.f);
        for (int p = 0; p < P; ++p)
            for (int k = 0; k < K; ++k) {
                int32_t id = (int32_t) (rng() % V);
                std::memcpy(&lp[(size_t) p * basalt::kernels::kLogprobStride + 2 + 2 * k], &id, 4);
                lp[(size_t) p * basalt::kernels::kLogprobStride + 3 + 2 * k] = -0.3f * k + 0.1f * nd(rng);
            }
        const int32_t anchor = 17;
        float *dlp = up(lp), *dh = up(h), *dconf = nullptr;
        uint16_t *dp = up(pred), *ds = up(succ);
        int32_t *da = up(std::vector<int32_t>{anchor}), *dids = nullptr;
        cudaMalloc(&dids, P * 4);
        cudaMalloc(&dconf, P * 4);
        basalt::kernels::candidate_path(dlp, basalt::kernels::kLogprobStride, K, dh, dp, ds, R, da, P, dids, dconf, {}, S);
        auto ids = down(dids, P);
        auto conf = down(dconf, P);
        int32_t prev = anchor;
        int differ = 0;
        std::vector<float> gc;
        std::vector<double> wc;
        for (int p = 0; p < P; ++p) {
            double best = -1e300, z = 0;
            int bk = 0;
            std::vector<double> sc(K);
            for (int k = 0; k < K; ++k) {
                int32_t c;
                std::memcpy(&c, &lp[(size_t) p * basalt::kernels::kLogprobStride + 2 + 2 * k], 4);
                double d = 0;
                for (int r = 0; r < R; ++r) d += (double) fb(pred[(size_t) prev * R + r]) * h[(size_t) p * R + r] * fb(succ[(size_t) c * R + r]);
                sc[(size_t) k] = lp[(size_t) p * basalt::kernels::kLogprobStride + 3 + 2 * k] + d;
                if (sc[(size_t) k] > best) { best = sc[(size_t) k]; bk = k; }
            }
            for (double v : sc) z += std::exp(v - best);
            std::memcpy(&prev, &lp[(size_t) p * basalt::kernels::kLogprobStride + 2 + 2 * bk], 4);
            differ += prev != ids[(size_t) p];
            prev = ids[(size_t) p];   // follow the GPU's path so one tie does not cascade
            gc.push_back(conf[(size_t) p]);
            wc.push_back(1.0 / z);
        }
        std::printf("candidate_path          picks differing from the CPU walk: %d of %d\n", differ, P);
        if (differ) ++fails;
        report("candidate_path conf", gc, wc, 1e-4);
    }
    {   // gqa_window_attn at the drafter's shape: 8 non-causal rows over a 2048-cell window in a cyclic cache
        const int NH = 32, NKV = 8, HD = 128, W = 2048, CAP = W + 256, B = 8;
        const int64_t pos = 5000;   // rows at 5000..5007; the cache wraps
        std::vector<uint16_t> kc((size_t) CAP * NKV * HD), vc(kc.size());
        for (auto& v : kc) v = basalt::kernels::f16_from_f32(nd(rng));
        for (auto& v : vc) v = basalt::kernels::f16_from_f32(nd(rng));
        auto q = randn((size_t) B * NH * HD);
        float* dq = up(q);
        uint16_t *dk = up(kc), *dv = up(vc);
        float *dout = nullptr, *dscr = nullptr;
        cudaMalloc(&dout, (size_t) B * NH * HD * 4);
        cudaMalloc(&dscr, basalt::kernels::gqa_decode_scratch_floats(CAP, B, NH, HD) * 4);
        basalt::kernels::gqa_window_attn(dq, dk, dv, NH, NKV, HD, pos, B, pos + B, W, CAP, dscr, CAP, dout, S);
        auto out = down(dout, (size_t) B * NH * HD);
        std::vector<double> want(out.size());
        for (int t = 0; t < B; ++t)
            for (int h = 0; h < NH; ++h) {
                const int kvh = h / (NH / NKV);
                const int64_t lo = pos + t - W + 1, hi = pos + B;
                std::vector<double> s;
                double mx = -1e300;
                for (int64_t p = lo; p < hi; ++p) {
                    double d = 0;
                    for (int i = 0; i < HD; ++i)
                        d += (double) q[((size_t) t * NH + h) * HD + i] * basalt::kernels::f32_from_f16(kc[(((size_t) (p % CAP)) * NKV + kvh) * HD + i]);
                    s.push_back(d / std::sqrt((double) HD));
                    mx = std::max(mx, s.back());
                }
                double l = 0;
                for (auto& v : s) { v = std::exp(v - mx); l += v; }
                for (int i = 0; i < HD; ++i) {
                    double a = 0;
                    for (size_t j = 0; j < s.size(); ++j)
                        a += s[j] * basalt::kernels::f32_from_f16(vc[(((size_t) ((lo + (int64_t) j) % CAP)) * NKV + kvh) * HD + i]);
                    want[((size_t) t * NH + h) * HD + i] = a / l;
                }
            }
        report("gqa_window_attn", out, want, 2e-4);
        // fused plain BF16 output via gqa_window_attn_segs (out=nullptr, out16!=nullptr) must be bitwise bf(out)
        uint16_t* dout16 = nullptr;
        cudaMalloc(&dout16, (size_t) B * NH * HD * 2);
        basalt::kernels::GqaWinSeg seg{B, dk, dv, pos, pos + B, W, CAP, nullptr};
        basalt::kernels::gqa_window_attn_segs(dq, NH, NKV, HD, &seg, 1, dscr, CAP, nullptr, S, dout16);
        auto out16 = down(dout16, (size_t) B * NH * HD);
        int gqa16_diff = 0;
        for (size_t i = 0; i < out.size(); ++i)
            if (out16[i] != bf(out[i])) ++gqa16_diff;
        std::printf("gqa_window_attn_segs out16 bitwise diffs: %d%s\n", gqa16_diff, gqa16_diff ? "  FAIL" : "");
        if (gqa16_diff) ++fails;
        cudaFree(dout16);
        // INT8 cyclic window attention (HD=128) via gqa_window_attn_segs (both FP32 and BF16 out16, with win_dev)
        basalt::kernels::KvInt8 q8;
        const size_t ce = (size_t) CAP * NKV * HD;
        cudaMalloc(&q8.k, ce); cudaMalloc(&q8.v, ce);
        cudaMalloc(&q8.ks, ce / 64 * 2); cudaMalloc(&q8.vs, ce / 64 * 2);
        std::vector<int8_t> hk(ce), hv(ce);
        std::vector<uint16_t> hks(ce / 64), hvs(ce / 64);
        for (size_t g = 0; g < ce / 64; ++g) {
            float mk = 0.0f, mv = 0.0f;
            for (size_t i = 0; i < 64; ++i) {
                mk = std::max(mk, std::fabs(basalt::kernels::f32_from_f16(kc[g * 64 + i])));
                mv = std::max(mv, std::fabs(basalt::kernels::f32_from_f16(vc[g * 64 + i])));
            }
            hks[g] = basalt::kernels::f16_from_f32(mk / 127.0f);
            hvs[g] = basalt::kernels::f16_from_f32(mv / 127.0f);
            const float sk = basalt::kernels::f32_from_f16(hks[g]), sv = basalt::kernels::f32_from_f16(hvs[g]);
            for (size_t i = 0; i < 64; ++i) {
                const float fk = basalt::kernels::f32_from_f16(kc[g * 64 + i]);
                const float fv = basalt::kernels::f32_from_f16(vc[g * 64 + i]);
                hk[g * 64 + i] = (int8_t) std::clamp(std::lrintf(sk > 0 ? fk / sk : 0.0f), -127l, 127l);
                hv[g * 64 + i] = (int8_t) std::clamp(std::lrintf(sv > 0 ? fv / sv : 0.0f), -127l, 127l);
            }
        }
        cudaMemcpy(q8.k, hk.data(), ce, cudaMemcpyHostToDevice);
        cudaMemcpy(q8.v, hv.data(), ce, cudaMemcpyHostToDevice);
        cudaMemcpy(q8.ks, hks.data(), ce / 64 * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(q8.vs, hvs.data(), ce / 64 * 2, cudaMemcpyHostToDevice);
        const int32_t win_h[4] = {(int32_t) pos, (int32_t) (pos + B), (int32_t) W, (int32_t) (std::max<int64_t>(0, pos - W + 1) / basalt::kernels::kGqaChunk)};
        int32_t* dwin = nullptr;
        cudaMalloc(&dwin, sizeof(win_h));
        cudaMemcpy(dwin, win_h, sizeof(win_h), cudaMemcpyHostToDevice);
        basalt::kernels::GqaWinSeg seg8{B, nullptr, nullptr, 0, 0, 0, CAP, dwin, q8};
        basalt::kernels::gqa_window_attn_segs(dq, NH, NKV, HD, &seg8, 1, dscr, CAP, dout, S);
        auto out8 = down(dout, (size_t) B * NH * HD);
        std::vector<double> want8(out8.size());
        for (int t = 0; t < B; ++t)
            for (int h = 0; h < NH; ++h) {
                const int kvh = h / (NH / NKV);
                const int64_t lo = pos + t - W + 1, hi = pos + B;
                std::vector<double> s;
                double mx = -1e300;
                for (int64_t p = lo; p < hi; ++p) {
                    const size_t row = (size_t) (p % CAP) * NKV + kvh;
                    double d = 0;
                    for (int i = 0; i < HD; ++i) {
                        const float sk = basalt::kernels::f32_from_f16(hks[row * 2 + i / 64]);
                        d += (double) q[((size_t) t * NH + h) * HD + i] * (double) ((float) hk[row * HD + i] * sk);
                    }
                    s.push_back(d / std::sqrt((double) HD));
                    mx = std::max(mx, s.back());
                }
                double l = 0;
                for (auto& v : s) { v = std::exp(v - mx); l += v; }
                for (int i = 0; i < HD; ++i) {
                    double a = 0;
                    for (size_t j = 0; j < s.size(); ++j) {
                        const size_t row = (size_t) ((lo + (int64_t) j) % CAP) * NKV + kvh;
                        const float sv = basalt::kernels::f32_from_f16(hvs[row * 2 + i / 64]);
                        a += s[j] * (double) ((float) hv[row * HD + i] * sv);
                    }
                    want8[((size_t) t * NH + h) * HD + i] = a / l;
                }
            }
        report("gqa_window_attn_segs i8", out8, want8, 2e-4);
        uint16_t* dout8_16 = nullptr;
        cudaMalloc(&dout8_16, (size_t) B * NH * HD * 2);
        basalt::kernels::gqa_window_attn_segs(dq, NH, NKV, HD, &seg8, 1, dscr, CAP, nullptr, S, dout8_16);
        auto out8_16 = down(dout8_16, (size_t) B * NH * HD);
        int gqa8_16_diff = 0;
        for (size_t i = 0; i < out8.size(); ++i)
            if (out8_16[i] != bf(out8[i])) ++gqa8_16_diff;
        std::printf("gqa_window_attn_segs i8 out16 bitwise diffs: %d%s\n", gqa8_16_diff, gqa8_16_diff ? "  FAIL" : "");
        if (gqa8_16_diff) ++fails;
        cudaFree(dout8_16); cudaFree(dwin);
        cudaFree(q8.k); cudaFree(q8.v); cudaFree(q8.ks); cudaFree(q8.vs);
    }
    {   // attn_qk_prepare_segs (HD=256 gated & HD=128 ungated, FP16 and INT8): bitwise vs per-segment attn_qk_prepare
        for (const int hd : {256, 128}) {
            const bool gated = hd == 256;
            const int NH = hd == 256 ? 24 : 32, NKV = hd == 256 ? 4 : 8, qs = gated ? 2 * hd : hd;
            const int lens[3] = {3, 5, 2}, R = 10;
            const int64_t cap = hd == 128 ? 64 : 0;
            const int64_t c0[3] = {10, 60, 25};
            basalt::kernels::RopeSpec rope;
            rope.rot = hd == 256 ? 64 : 128;
            rope.theta = 1e6f;
            auto qf = randn((size_t) R * NH * qs), kk = randn((size_t) R * NKV * hd), vv = randn((size_t) R * NKV * hd);
            auto qn = randn(hd, 0.2f), kn = randn(hd, 0.2f);
            for (auto& v : qn) v += 1.0f;
            for (auto& v : kn) v += 1.0f;
            std::vector<int32_t> pos((size_t) R * 3);
            for (int t = 0; t < R; ++t) pos[t * 3] = pos[t * 3 + 1] = pos[t * 3 + 2] = 100 + t;
            float *dqf = up(qf), *dk = up(kk), *dv = up(vv), *dqn = up(qn), *dkn = up(kn);
            int32_t* dpos = up(pos);
            float *qo_ref = nullptr, *qo_seg = nullptr;
            cudaMalloc(&qo_ref, (size_t) R * NH * hd * 4);
            cudaMalloc(&qo_seg, (size_t) R * NH * hd * 4);
            const size_t cache_cells = 128;
            const size_t ce = cache_cells * NKV * hd;
            for (const bool i8 : {false, true}) {
                uint16_t *kr[3] = {}, *vr[3] = {}, *ks[3] = {}, *vs[3] = {};
                basalt::kernels::KvInt8 q8r[3] = {}, q8s[3] = {};
                basalt::kernels::QkPrepSeg segs[3] = {};
                int row = 0;
                for (int i = 0; i < 3; ++i) {
                    if (!i8) {
                        cudaMalloc(&kr[i], ce * 2); cudaMalloc(&vr[i], ce * 2);
                        cudaMalloc(&ks[i], ce * 2); cudaMalloc(&vs[i], ce * 2);
                        cudaMemset(kr[i], 0, ce * 2); cudaMemset(vr[i], 0, ce * 2);
                        cudaMemset(ks[i], 0, ce * 2); cudaMemset(vs[i], 0, ce * 2);
                    } else {
                        cudaMalloc(&q8r[i].k, ce); cudaMalloc(&q8r[i].v, ce);
                        cudaMalloc(&q8r[i].ks, ce / 64 * 2); cudaMalloc(&q8r[i].vs, ce / 64 * 2);
                        cudaMalloc(&q8s[i].k, ce); cudaMalloc(&q8s[i].v, ce);
                        cudaMalloc(&q8s[i].ks, ce / 64 * 2); cudaMalloc(&q8s[i].vs, ce / 64 * 2);
                        cudaMemset(q8r[i].k, 0, ce); cudaMemset(q8r[i].v, 0, ce);
                        cudaMemset(q8r[i].ks, 0, ce / 64 * 2); cudaMemset(q8r[i].vs, 0, ce / 64 * 2);
                        cudaMemset(q8s[i].k, 0, ce); cudaMemset(q8s[i].v, 0, ce);
                        cudaMemset(q8s[i].ks, 0, ce / 64 * 2); cudaMemset(q8s[i].vs, 0, ce / 64 * 2);
                    }
                    basalt::kernels::attn_qk_prepare(dqf + (size_t) row * NH * qs, dk + (size_t) row * NKV * hd,
                                                     dv + (size_t) row * NKV * hd, dqn, dkn, 1e-6f, rope,
                                                     dpos + (size_t) row * 3, lens[i], NH, NKV,
                                                     qo_ref + (size_t) row * NH * hd, kr[i], vr[i], c0[i], S, hd, gated, cap,
                                                     nullptr, q8r[i]);
                    segs[i] = {lens[i], ks[i], vs[i], c0[i], cap, nullptr, q8s[i], {}};
                    row += lens[i];
                }
                basalt::kernels::attn_qk_prepare_segs(dqf, dk, dv, dqn, dkn, 1e-6f, rope, dpos, segs, 3, NH, NKV, qo_seg, S, hd, gated);
                auto hqr = down(qo_ref, (size_t) R * NH * hd), hqs = down(qo_seg, (size_t) R * NH * hd);
                int diff = std::memcmp(hqr.data(), hqs.data(), hqr.size() * 4) != 0;
                for (int i = 0; i < 3; ++i) {
                    if (!i8) {
                        auto a = down(kr[i], ce), b = down(ks[i], ce), c = down(vr[i], ce), d = down(vs[i], ce);
                        diff += (std::memcmp(a.data(), b.data(), ce * 2) != 0) + (std::memcmp(c.data(), d.data(), ce * 2) != 0);
                        cudaFree(kr[i]); cudaFree(vr[i]); cudaFree(ks[i]); cudaFree(vs[i]);
                    } else {
                        auto a = down(q8r[i].k, ce), b = down(q8s[i].k, ce), c = down(q8r[i].v, ce), d = down(q8s[i].v, ce);
                        auto as = down(q8r[i].ks, ce / 64), bs = down(q8s[i].ks, ce / 64), cs = down(q8r[i].vs, ce / 64), ds = down(q8s[i].vs, ce / 64);
                        diff += (std::memcmp(a.data(), b.data(), ce) != 0) + (std::memcmp(c.data(), d.data(), ce) != 0) +
                                (std::memcmp(as.data(), bs.data(), ce / 64 * 2) != 0) + (std::memcmp(cs.data(), ds.data(), ce / 64 * 2) != 0);
                        cudaFree(q8r[i].k); cudaFree(q8r[i].v); cudaFree(q8r[i].ks); cudaFree(q8r[i].vs);
                        cudaFree(q8s[i].k); cudaFree(q8s[i].v); cudaFree(q8s[i].ks); cudaFree(q8s[i].vs);
                    }
                }
                std::printf("attn_qk_prepare_segs hd=%d %s bitwise diffs: %d%s\n", hd, i8 ? "int8" : "fp16", diff, diff ? "  FAIL" : "");
                if (diff) ++fails;
            }
            cudaFree(dqf); cudaFree(dk); cudaFree(dv); cudaFree(dqn); cudaFree(dkn); cudaFree(dpos); cudaFree(qo_ref); cudaFree(qo_seg);
        }
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
