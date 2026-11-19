// include/basalt/platform/compiler.hpp - the compiler- and CPU-specific spellings the host code needs, behind
// one name each, so no source file asks which compiler it is built with.
//
// The engine's CPU kernels are AVX2 / AVX-512 (x86-64). On ARM64 (Windows on Arm, aarch64 Linux) the same names
// exist so the shared host code builds: CPUID reads as all zeros, so every `cpu_*_ok()` probe says no and the x86
// kernels are never chosen (their files are replaced by src/kernels/cpu/arm64_stubs.cpp). Every function is inline
// and maps to the one instruction or builtin each compiler gives for it.
#pragma once

#include <cstdint>

#if defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__aarch64__)
#define BASALT_HOST_ARM64 1
#else
#define BASALT_HOST_ARM64 0
#endif

#if BASALT_HOST_ARM64
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <arm_neon.h>
#else
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// Marks a function that uses AVX2, FMA and F16C intrinsics in a translation unit built without those flags. MSVC
// accepts the intrinsics in any function and needs no marking.
#if defined(_MSC_VER)
#define BASALT_TARGET_AVX2_FMA_F16C
#else
#define BASALT_TARGET_AVX2_FMA_F16C __attribute__((target("avx2,fma,f16c")))
#endif

namespace basalt::platform {

// MSVC can retain temporary stack copies for vector aggregate assignments even after inlining.
// Field-wise assignment exposes the values to register allocation.
#if defined(_MSC_VER) && !defined(__clang__)
inline constexpr bool kPreferMemberwiseVectorAssignment = true;
#else
inline constexpr bool kPreferMemberwiseVectorAssignment = false;
#endif

/// CPUID leaf `leaf`, subleaf `sub`: r[0..3] = EAX, EBX, ECX, EDX. All zeros on ARM64 (no x86 feature is present).
inline void cpuid(unsigned leaf, unsigned sub, unsigned (&r)[4]) {
#if BASALT_HOST_ARM64
    (void) leaf;
    (void) sub;
    r[0] = r[1] = r[2] = r[3] = 0;
#elif defined(_MSC_VER)
    int x[4];
    __cpuidex(x, (int) leaf, (int) sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned) x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

/// XCR0, which says which register state the OS saves (read only after CPUID reports OSXSAVE).
inline unsigned long long xgetbv0() {
#if BASALT_HOST_ARM64
    return 0;
#elif defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long) hi << 32) | lo;
#endif
}

/// The index of the lowest set bit of a nonzero mask.
inline int ctz32(unsigned m) {
#if defined(_MSC_VER)
    unsigned long i = 0;
    _BitScanForward(&i, m);
    return (int) i;
#else
    return __builtin_ctz(m);
#endif
}

/// One iteration of a spin wait: yields the pipeline to the core's SMT sibling (`pause` on x86, `yield` on ARM64).
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif BASALT_HOST_ARM64 && defined(_MSC_VER)
    __yield();
#elif BASALT_HOST_ARM64
    __asm__ __volatile__("yield");
#endif
}

/// A prefetch of the cache line at p into L2 (`prefetcht1` on x86, `prfm pldl2keep` on ARM64).
inline void prefetch_l2(const void* p) {
#if BASALT_HOST_ARM64 && defined(_MSC_VER)
    __prefetch2(p, 2);   // PLD, L2, KEEP
#elif BASALT_HOST_ARM64
    __builtin_prefetch(p, 0, 2);
#else
    _mm_prefetch((const char*) p, _MM_HINT_T1);
#endif
}

/// The fp32 -> half -> fp32 round trip, round to nearest even. Where the compiler has a half type it does the
/// conversion; MSVC has none, and its F16C intrinsic pair is the same instruction with the same rounding.
inline float f16_round(float x) {
#if BASALT_HOST_ARM64
    return vgetq_lane_f32(vcvt_f32_f16(vcvt_f16_f32(vdupq_n_f32(x))), 0);   // FCVTN / FCVTL, round to nearest even
#elif defined(_MSC_VER)
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
#else
    return (float) (_Float16) x;
#endif
}

/// Whether this CPU, and the OS saving its AVX-512 state, can run AVX512_BF16 with AVX512BW.
inline bool cpu_has_avx512_bf16() {
#if BASALT_HOST_ARM64
    return false;
#elif defined(_MSC_VER)
    // The same readings GCC's run-time probe takes: the OS has to save the AVX-512 state, and leaf 7 subleaf 0
    // names AVX512BW (EBX bit 30) and AVX512_BF16 (ECX bit 5).
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] < 7u) return false;
    cpuid(1, 0, r);
    if (!((r[2] >> 27) & 1u)) return false;                 // OSXSAVE
    if ((xgetbv0() & 0xE6) != 0xE6) return false;           // the OS saves the AVX-512 state
    cpuid(7, 0, r);
    return ((r[1] >> 30) & 1u) != 0 && ((r[2] >> 5) & 1u) != 0;   // AVX512BW, AVX512_BF16
#else
    return __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512bw");
#endif
}

}  // namespace basalt::platform
