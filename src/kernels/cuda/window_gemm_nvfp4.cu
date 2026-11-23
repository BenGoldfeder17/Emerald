// src/kernels/cuda/window_gemm_nvfp4.cu - the decode window products against NVFP4 matrices in planes, alone and as a
// SwiGLU pair (window_gemm.cuh).
// Adapted from Infernix (Ian Ranson and contributors), Apache-2.0: see THIRD_PARTY_LICENSES.
#include "window_gemm_kernels.hpp"

namespace basalt::kernels::wgemm::detail {
namespace {
constexpr int F = (int) Format::Nvfp4, CH = 256, MW = 1;
}

const Kernel& nvfp4(int nt) {
    static const Kernel k1 = make_kernel<F, 64, 1, CH, 4, 1, MW>(), k2 = make_kernel<F, 64, 1, CH, 4, 2, MW>(),
                        k3 = make_kernel<F, 64, 1, CH, 4, 3, MW>(), k4 = make_kernel<F, 64, 1, CH, 4, 4, MW>();
    return nt > 3 ? k4 : nt > 2 ? k3 : nt > 1 ? k2 : k1;
}

const Kernel& nvfp4_swiglu(int nt) {
    // NT == 1 (148 regs) fits 3 blocks/SM with ST = 3; NT >= 2 (186-222 regs) is register-capped at 2 blocks/SM so keeps ST = 4
    static const Kernel k1 = make_kernel<F, 32, 2, CH, 3, 1, MW>(), k2 = make_kernel<F, 32, 2, CH, 4, 2, MW>(),
                        k3 = make_kernel<F, 32, 2, CH, 4, 3, MW>(), k4 = make_kernel<F, 32, 2, CH, 4, 4, MW>();
    return nt > 3 ? k4 : nt > 2 ? k3 : nt > 1 ? k2 : k1;
}

}  // namespace basalt::kernels::wgemm::detail
