// src/kernels/cpu/arm64_stubs.cpp - ARM64 builds only: stands in for the x86-only CPU kernel files (AVX2,
// AVX-512, AVX-VNNI; see CMakeLists.txt, kernels_cpu). On ARM64 platform::cpuid reads as zeros, so every
// cpu_*_ok() probe is false and no caller selects these kernels; the predicates below say "not supported" and
// the kernels stop the process with a message if anything still reaches them. The CPU expert tier (MoE models'
// experts in RAM) is therefore unavailable on ARM64; the dense models keep every weight on the GPU.
#include <cstdio>
#include <cstdlib>

// the stubs keep the declared parameter names, which they do not read
#if defined(_MSC_VER)
#pragma warning(disable : 4100)
#else
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/iq_avx2.hpp"
#include "basalt/kernels/cpu/iq_avx512.hpp"
#include "basalt/kernels/cpu/kq_avx1.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx2.hpp"
#include "basalt/kernels/cpu/nvfp4_avx512.hpp"

namespace basalt::kernels::cpu {
namespace {
[[noreturn]] void x86_only(const char* fn) {
    std::fprintf(stderr, "basalt: %s is an x86 (AVX2/AVX-512) CPU kernel and is not built for ARM64\n", fn);
    std::fflush(stderr);
    std::abort();
}
}  // namespace

const char* CpuFeatures::reason() const { return usable() ? "ok" : "ARM64 CPU: the AVX-512 expert kernel is x86-only"; }
CpuFeatures cpu_features() { return CpuFeatures{}; }

void act_quant_q8_1(const float* x, int n, ActQ& a) { x86_only("act_quant_q8_1"); }
void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out, int r0, int r1) { x86_only("q2_0_gguf_rows_multi"); }
void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out, int r0, int r1) { x86_only("q2_0_gguf_rows_multi_avx2"); }
void q2_0_gguf_rows_multi_avx2_base(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out, int r0, int r1) { x86_only("q2_0_gguf_rows_multi_avx2_base"); }
void q2_0_gguf_rows_multi_avx2_vnni(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out, int r0, int r1) { x86_only("q2_0_gguf_rows_multi_avx2_vnni"); }
void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) { x86_only("act_quant_q8_1_avx2"); }
void quantize_q8_K_avx2(const float* x, void* dst, int64_t k) { x86_only("quantize_q8_K_avx2"); }
bool iq256_supported(int ggml_type) noexcept { return false; }
void iq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("iq256_gu_rows"); }
void iq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("iq256_rows"); }
void iq256_gu_rows_base(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("iq256_gu_rows_base"); }
void iq256_gu_rows_vnni(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("iq256_gu_rows_vnni"); }
void iq256_rows_base(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("iq256_rows_base"); }
void iq256_rows_vnni(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("iq256_rows_vnni"); }
void iq256_gu_rows_mode_base(int mode, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("iq256_gu_rows_mode_base"); }
void iq256_gu_rows_mode_vnni(int mode, int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("iq256_gu_rows_mode_vnni"); }
void iq256_rows_mode_base(int mode, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("iq256_rows_mode_base"); }
void iq256_rows_mode_vnni(int mode, int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("iq256_rows_mode_vnni"); }
bool iq4nl256_fits(int n) noexcept { return false; }
void iq4nl256_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out, int r0, int r1) { x86_only("iq4nl256_down_rows"); }
void iq4nl256_down_rows_base(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out, int r0, int r1) { x86_only("iq4nl256_down_rows_base"); }
void iq4nl256_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, const void* const* hq, int nt, float* const* out, int r0, int r1) { x86_only("iq4nl256_down_rows_vnni"); }
void quantize_q8_K_avx512(const float* x, void* dst, int64_t k) { x86_only("quantize_q8_K_avx512"); }
bool iq512_supported(int ggml_type) noexcept { return false; }
void iq512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("iq512_gu_rows"); }
void iq512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("iq512_rows"); }
bool iq1s_512_fits(int n) noexcept { return false; }
void iq1s_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("iq1s_512_gu_rows"); }
void iq1s_512_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("iq1s_512_rows"); }
bool iq_exact_512_supported(int ggml_type) noexcept { return false; }
void iq_exact_512_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, bool scaled, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("iq_exact_512_gu_rows"); }
void iq_exact_512_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("iq_exact_512_rows"); }
void bf16_rows_dot_multi_avx1(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) { x86_only("bf16_rows_dot_multi_avx1"); }
bool kq256_supported(int ggml_type) noexcept { return false; }
void kq256_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("kq256_gu_rows"); }
void kq256_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("kq256_rows"); }
void bf16_rows_dot_multi(const uint16_t* w, int rows, int cols, const float* x, int nt, float* out) { x86_only("bf16_rows_dot_multi"); }
bool nvfp4_256_fits(int n_embd, int n_ff) noexcept { return false; }
void nvfp4_256_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("nvfp4_256_gu_rows"); }
void nvfp4_256_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("nvfp4_256_down_rows"); }
void nvfp4_256_gu_rows_base(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("nvfp4_256_gu_rows_base"); }
void nvfp4_256_gu_rows_vnni(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("nvfp4_256_gu_rows_vnni"); }
void nvfp4_256_down_rows_base(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("nvfp4_256_down_rows_base"); }
void nvfp4_256_down_rows_vnni(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("nvfp4_256_down_rows_vnni"); }
bool nvfp4_512_fits(int n_embd, int n_ff) noexcept { return false; }
void nvfp4_512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, float sg, float su, float limit, const void* const* act, int nt, float* const* ff, int r0, int r1) { x86_only("nvfp4_512_gu_rows"); }
void nvfp4_512_down_rows(const uint8_t* w, size_t row_bytes, int n, float sd, const void* const* act, int nt, float* const* out, int r0, int r1) { x86_only("nvfp4_512_down_rows"); }

}  // namespace basalt::kernels::cpu

// The extracted ggml CPU dot products need AVX2 (ggml_cpu_dots.c): no table on ARM64.
extern "C" const ggml_type_traits_cpu* ggml_get_type_traits_cpu(ggml_type) { return nullptr; }
