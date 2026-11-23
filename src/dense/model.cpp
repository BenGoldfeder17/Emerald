// src/dense/model.cpp - the dense stack's shared scratch and geometry checks.
#include "internal.hpp"

#include "basalt/kernels/gdn_prompt.hpp"
#include "basalt/kernels/gqa_attn.hpp"
#include "basalt/kernels/logprobs.hpp"
#include "basalt/kernels/nvfp4_planes.hpp"
#include "basalt/prefill/fp4_dense.hpp"
#include "basalt/prefill/fp8_gemm.hpp"
#include "basalt/kernels/native_mmvq.hpp"
#include "basalt/log.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/mmproj.hpp"

#include <algorithm>
#include <cstdlib>

namespace basalt::dense {

Arena::~Arena() {
    if (base_) cudaFree(base_);
}

bool Arena::reserve(size_t bytes) {
    if (base_) cudaFree(base_);
    base_ = nullptr;
    size_ = used_ = 0;
    if (bytes == 0) return true;
    if (cudaMalloc(&base_, bytes) != cudaSuccess) {
        cudaGetLastError();
        base_ = nullptr;
        return false;
    }
    size_ = bytes;
    return true;
}

void* Arena::take_bytes(size_t n) {
    const size_t at = (used_ + 255) & ~(size_t) 255;
    if (at + n > size_) return nullptr;
    used_ = at + n;
    return base_ + at;
}

int Geometry::gdn_layers() const { return (int) std::count(kinds.begin(), kinds.end(), Mixer::Gdn); }
int Geometry::attn_layers() const { return (int) std::count(kinds.begin(), kinds.end(), Mixer::Attention); }

std::string Geometry::check() const {
    if ((int) kinds.size() != n_layers || n_layers <= 0) return "the layer kinds do not cover the layers";
    if (n_embd % 256 || n_ff % 256) return "widths must be whole 256-value blocks";
    if (attn_layers() > 0) {
        if (head_dim != 256) return "attention heads must be 256 wide";
        if (n_kv <= 0 || n_head % n_kv || n_head / n_kv > 8) return "attention head grouping not supported";
        if (rope.rot <= 0 || rope.rot > head_dim || rope.rot % 2 || rope.rot / 2 > 64) return "rotary width not supported";
    }
    if (gdn_layers() > 0) {
        if (!kernels::gdn_prompt_supported(gdn_state, gdn_k_heads, gdn_v_heads, gdn_conv)) return "GDN geometry not supported";
        if (!kernels::dec_gdn_rec_supported(n_embd)) return "GDN window kernel not built for this width";
    }
    return {};
}

namespace {

// bytes of the decode window's scratch
// the block drafter's rows in the window scratch
size_t block_scratch_floats(const Geometry& g, const BlockDrafter& bd) {
    if (!bd.present) return 0;
    const size_t W = kBatch, AQ = (size_t) bd.n_head * bd.head_dim, KV = (size_t) bd.n_kv * bd.head_dim;
    return W * (3 * AQ + 2 * KV + bd.layers[0].conv_a.rows + g.n_embd + kernels::kLogprobStride + bd.rank + 3 + 4) +
           kernels::gqa_decode_scratch_floats(bd.window + 256, kBatch, bd.n_head, bd.head_dim) + 64 * 64;
}

int64_t widest_input(const Geometry& g) {
    const size_t AQ = (size_t) g.n_head * g.head_dim;
    return (int64_t) std::max<size_t>({(size_t) g.n_embd, (size_t) g.n_ff, (size_t) g.value_width(), AQ, 2 * (size_t) g.n_embd});
}
// a window's activation buffers for the widest input: the window gemm's bf16 rows, MMQ's q8_1 rows
size_t act_slot(const Geometry& g) { return (kernels::wgemm::act_bytes(widest_input(g)) + 255) / 256 * 256; }
size_t q8_slot(const Geometry& g) { return (prefill::mmq::q8_bytes(kBatch, widest_input(g)) + 255) / 256 * 256; }

// The window gemm's split-K workspace for every call a window makes (forward.cpp): the groups and each matrix alone, in
// either format (a superset is harmless).
void wgemm_need(const Geometry& g, size_t& partial, size_t& tickets) {
    namespace wg = kernels::wgemm;
    const int64_t H = g.n_embd, AQ = (int64_t) g.n_head * g.head_dim, KV = (int64_t) g.n_kv * g.head_dim;
    const int64_t C = g.conv_channels(), VD = g.value_width(), F = g.n_ff;
    struct Shape {
        int64_t rows[3];
        int n;
        int64_t K;
        wg::Epilogue epi;
    };
    std::vector<Shape> shapes = {{{2 * AQ, KV, KV}, 3, H, wg::Epilogue::Store}, {{2 * AQ}, 1, H, wg::Epilogue::Store},
                                 {{KV}, 1, H, wg::Epilogue::Store},          {{H}, 1, AQ, wg::Epilogue::Add},
                                 {{F, F}, 2, H, wg::Epilogue::SwiGLU},       {{F}, 1, H, wg::Epilogue::Store},
                                 {{H}, 1, F, wg::Epilogue::Add},             {{g.n_vocab}, 1, H, wg::Epilogue::Store},
                                 {{256}, 1, H, wg::Epilogue::Store},   // the block drafter's selector (its rank)
                                 {{kDraftVocab}, 1, H, wg::Epilogue::Store}};   // and its head's rows
    if (g.gdn_layers() > 0) {
        shapes.push_back({{C, VD}, 2, H, wg::Epilogue::Store});
        shapes.push_back({{C}, 1, H, wg::Epilogue::Store});
        shapes.push_back({{VD}, 1, H, wg::Epilogue::Store});
        shapes.push_back({{H}, 1, VD, wg::Epilogue::Add});
    }
    partial = tickets = 0;
    for (const Shape& sh : shapes)
        for (const wg::Format f : {wg::Format::F8, wg::Format::Nvfp4, wg::Format::Bf16}) {
            if (sh.K <= 0 || sh.K % 256 != 0) continue;
            wg::Call c;
            c.fmt = f;
            c.K = sh.K;
            c.n = sh.n;
            c.epi = sh.epi;
            bool ok = true;
            for (int i = 0; i < sh.n; ++i) {
                c.m[i].rows = sh.rows[i];
                ok = ok && sh.rows[i] > 0 && (f == wg::Format::F8 || sh.rows[i] % 128 == 0);
            }
            if (!ok || (f != wg::Format::Nvfp4 && sh.epi == wg::Epilogue::SwiGLU)) continue;
            size_t p, t;
            wg::workspace_need(c, p, t);
            partial = std::max(partial, p);
            tickets = std::max(tickets, t);
        }
}

size_t window_bytes(const Geometry& g, const BlockDrafter& bd, int64_t max_context, int n_gdn = -1, int rec_rows = kBatch) {
    const size_t W = kBatch, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim;
    const size_t ng = (size_t) (n_gdn >= 0 ? n_gdn : g.gdn_layers());
    const size_t C = (size_t) g.conv_channels(), HKV = C - (size_t) g.gdn_k_heads * (size_t) g.gdn_state;
    size_t f = 4 * W * H + W * g.value_width() + W * (2 * AQ + 2 * (size_t) g.n_kv * g.head_dim + 3 * AQ) + 3 * W * g.n_ff;
    if (g.attn_layers() > 0) f += kernels::gqa_decode_scratch_floats(max_context, kBatch, g.n_head);
    f += 2 * W * (size_t) g.gdn_v_heads + (sizeof(BatchTable) + 3) / 4;
    if (ng > 0) f += ng * (size_t) rec_rows * (C + HKV + 2 * (size_t) g.gdn_v_heads);
    f += W * 2 * H + kernels::row_logprobs_scratch_floats(kBatch, g.n_vocab) + W * g.value_width() + W * (size_t) g.n_vocab;
    f += block_scratch_floats(g, bd);
    size_t wp, wt;
    wgemm_need(g, wp, wt);
    return f * 4 + 2 * act_slot(g) + q8_slot(g) + (wp + wt + 1) * 4 + W * 4 * 8 + 96 * 256;
}

struct ChunkPlan {
    bool gdn_bf16 = false, attn_bf16 = false, ffn_fp4 = false, any_fp4 = false;
    bool kv_int8 = false, mtp = false, block = false;
    size_t xq_bytes = 0, a4_bytes = 0, total_bytes = 0;
};

ChunkPlan plan_chunk(const Model& m) {
    const Geometry& gg = m.g;
    const size_t T = kChunk, H = (size_t) gg.n_embd, AQ = (size_t) gg.n_head * gg.head_dim, C = (size_t) gg.conv_channels();
    const size_t VD = (size_t) gg.value_width(), HV = (size_t) gg.gdn_v_heads, KV = (size_t) gg.n_kv * gg.head_dim;
    const size_t widest = std::max<size_t>({H, 2 * H, (size_t) gg.n_ff, VD, AQ});
    ChunkPlan p;
    p.kv_int8 = m.o.kv_int8;
    p.mtp = m.o.drafter == Drafter::Mtp;
    p.block = m.o.drafter == Drafter::Block;
    p.gdn_bf16 = gg.gdn_layers() > 0;
    p.attn_bf16 = gg.attn_layers() > 0;
    p.ffn_fp4 = true;
    size_t max_mmq_cols = 0, max_fp8_cols = 0;
    auto note = [&](const Linear& l) {
        if (!l.w) return;
        if (l.type == kTypeF8) max_fp8_cols = std::max(max_fp8_cols, (size_t) l.cols);
        else if (l.type == kTypeNvfp4Planes) p.any_fp4 = true;
        else if (l.type != kTypeQ8P) max_mmq_cols = std::max(max_mmq_cols, (size_t) l.cols);
    };
    for (int il = m.lb; il < m.le; ++il) {
        const Layer& L = m.w.layers[(size_t) il];
        if (L.kind == Mixer::Gdn) {
            const bool f8 = L.qkv.type == kTypeF8 && L.z.type == kTypeF8 && L.out.type == kTypeF8;
            const bool fp4 = L.qkv.type == kTypeNvfp4Planes && L.z.type == kTypeNvfp4Planes && L.out.type == kTypeNvfp4Planes;
            p.gdn_bf16 = p.gdn_bf16 && (f8 || fp4);
            note(L.qkv); note(L.z); note(L.out);
        } else {
            const bool f8 = L.q.type == kTypeF8 && L.k.type == kTypeF8 && L.v.type == kTypeF8 && L.o.type == kTypeF8;
            const bool fp4 = L.q.type == kTypeNvfp4Planes && L.k.type == kTypeNvfp4Planes && L.v.type == kTypeNvfp4Planes && L.o.type == kTypeNvfp4Planes;
            p.attn_bf16 = p.attn_bf16 && (f8 || fp4);
            note(L.q); note(L.k); note(L.v); note(L.o);
        }
        p.ffn_fp4 = p.ffn_fp4 && L.gate.type == kTypeNvfp4Planes && L.up.type == kTypeNvfp4Planes && L.down.type == kTypeNvfp4Planes;
        note(L.gate); note(L.up); note(L.down);
    }
    if (p.mtp) {
        p.attn_bf16 = false;
        p.ffn_fp4 = false;
        note(m.w.mtp.fc);
        note(m.w.mtp.layer.q); note(m.w.mtp.layer.k); note(m.w.mtp.layer.v); note(m.w.mtp.layer.o);
        note(m.w.mtp.layer.gate); note(m.w.mtp.layer.up); note(m.w.mtp.layer.down);
    }
    if (p.block) {
        for (const Linear& l : m.w.dflash.fc) note(l);
        for (const BlockDrafter::Layer& D : m.w.dflash.layers) { note(D.k); note(D.v); }
    }
    p.xq_bytes = std::max(max_mmq_cols ? prefill::mmq::q8_bytes(T, (int64_t) max_mmq_cols) : size_t(0),
                          max_fp8_cols ? prefill::fp8::act_bytes(T, (int64_t) max_fp8_cols) : size_t(0));
    p.a4_bytes = p.any_fp4 ? prefill::fp4_dense::act_bytes((int64_t) T, (int64_t) widest) : size_t(0);
    // keep buffers: BF16 residual x, FP32 xn/tmp, optional MTP cat/hn or Block dacc, xq, a4, gs, hmax, tok/pos/iota/bounds
    size_t keep = T * H * 2 + 2 * T * H * 4 + p.xq_bytes + p.a4_bytes + 2 * T * 4 + (5 * T + 2) * 4;
    if (p.mtp) keep += (2 * T + 1) * H * 4;
    if (p.block) keep += T * H * 4;
    const size_t gdn = gg.gdn_layers()
                           ? (p.gdn_bf16 ? (2 * T * C * 2 + 2 * T * VD * 2 + T * VD * 4 + 2 * T * HV * 4)
                                         : (2 * T * C + 3 * T * VD + 4 * T * HV) * 4) +
                                 kernels::gdn_work_bytes(T)
                           : 0;
    const size_t attn_buf = p.attn_bf16
                                ? (T * (2 * AQ + 2 * KV + AQ) * 2 + T * (2 * AQ) * (p.kv_int8 ? 2 : 4))
                                : T * (2 * AQ + 2 * KV + 3 * AQ) * 4;
    const size_t attn_sp = p.kv_int8 ? kernels::gqa_prompt_split_floats((int64_t) T, gg.n_head, kAttnSplits) * 4 : 0;
    const size_t attn = (gg.attn_layers() || p.mtp) ? attn_buf + attn_sp : 0;
    const size_t ffn = p.ffn_fp4 ? T * (size_t) gg.n_ff * 2 : 3 * T * (size_t) gg.n_ff * 4;
    const size_t blk = p.block ? 2 * T * (size_t) m.w.dflash.n_kv * (size_t) m.w.dflash.head_dim * 4 : 0;
    p.total_bytes = keep + std::max({gdn, attn, ffn, blk}) + 40 * 256;
    return p;
}

size_t chunk_bytes(const Geometry& g) {
    const size_t T = kChunk, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim, C = (size_t) g.conv_channels();
    const size_t VD = (size_t) g.value_width(), HV = (size_t) g.gdn_v_heads;
    const size_t widest = std::max<size_t>({H, (size_t) g.n_ff, VD, AQ});
    const size_t keep = 6 * T * H * 4 + H * 4 +
                        std::max(prefill::mmq::q8_bytes(T, (int64_t) widest), prefill::fp8::act_bytes(T, (int64_t) widest)) +
                        prefill::fp4_dense::act_bytes((int64_t) T, (int64_t) widest) + 2 * T * 4 + 5 * 256;
    const size_t gdn = g.gdn_layers() ? (2 * T * C + 3 * T * VD + 4 * T * HV) * 4 + kernels::gdn_work_bytes(T) : 0;
    const size_t attn = g.attn_layers() ? T * (2 * AQ + 2 * (size_t) g.n_kv * g.head_dim + 3 * AQ) * 4 +
                                              kernels::gqa_prompt_split_floats((int64_t) T, g.n_head, kAttnSplits) * 4
                                        : 0;
    const size_t ffn = 3 * T * g.n_ff * 4;
    return keep + std::max({gdn, attn, ffn}) + T * 4 * 6 + 32 * 256;
}

}  // namespace

namespace {
// The block drafter as decode windows run it, on the window gemm: q, o, gate, up and down rounded to NVFP4 planes in place
// and the convolutions repacked to Q8 planes in place (prompt chunks never read them); k and v (NVFP4) and the feature
// projection (Q8 planes), which prompt chunks read as blocks, get copies.  Then the window gemm's workspace grows to its
// calls if they need more.
bool decode_drafter(Model& m, std::string& err) {
    m.bd_dec = m.w.dflash;
    BlockDrafter& bd = m.bd_dec;
    if (!bd.present || m.o.drafter != Drafter::Block) return true;
    // the memory and the new forms now (later sizing sees them); the kernels that make them once the weights have landed
    // (attach_drafter), in this order, through one scratch buffer of the largest matrix
    std::vector<std::function<bool(void*)>>& steps = m.drafter_steps;
    size_t& tmp_bytes = m.drafter_tmp;
    auto q8 = [](const Linear& l) { return l.type == 8 && l.cols % 256 == 0 && l.rows > 0; };
    auto bytes = [](const Linear& l) { return (size_t) l.rows * (size_t) l.cols / 32 * 34; };
    auto in_place = [&](Linear& l) -> bool {
        if (!q8(l)) return true;
        const size_t n = bytes(l);
        tmp_bytes = std::max(tmp_bytes, n);
        void* w = const_cast<void*>(l.w);
        const int rows = l.rows, cols = l.cols;
        steps.push_back([=](void* tmp) {
            if (cudaMemcpy(tmp, w, n, cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
            kernels::wgemm::q8_planes(tmp, rows, cols, w, nullptr);
            return cudaDeviceSynchronize() == cudaSuccess;
        });
        l.type = kTypeQ8P;
        return true;
    };
    auto copy = [&](Linear& l) -> bool {
        if (!q8(l)) return true;
        void* p = nullptr;
        if (cudaMalloc(&p, bytes(l)) != cudaSuccess) return false;
        m.owned.push_back(p);
        const void* w = l.w;
        const int rows = l.rows, cols = l.cols;
        steps.push_back([=](void*) {
            kernels::wgemm::q8_planes(w, rows, cols, p, nullptr);
            return true;
        });
        l.w = p;
        l.type = kTypeQ8P;
        return true;
    };
    // The drafter's big products (gate, up, down, q, o) in NVFP4 planes, rounded from its Q8_0 at load: the drafts' precision
    // only (every draft is verified exactly, so the output does not change); about half the bytes a pass reads.
    auto to_fp4 = [&](Linear& l) -> bool {
        if (!q8(l) || !kernels::nvfp4p::fits(l.rows, l.cols)) return in_place(l);
        const size_t n = bytes(l);
        tmp_bytes = std::max(tmp_bytes, n);
        void* sc = nullptr;
        if (cudaMalloc(&sc, 16) != cudaSuccess) return false;
        m.owned.push_back(sc);
        void* w = const_cast<void*>(l.w);
        const int rows = l.rows, cols = l.cols;
        steps.push_back([=](void* tmp) {
            if (cudaMemcpy(tmp, w, n, cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
            kernels::nvfp4p::from_q8(tmp, rows, cols, w, (float*) sc, nullptr);
            return cudaDeviceSynchronize() == cudaSuccess;
        });
        l.scale = (const float*) sc;
        l.type = kTypeNvfp4Planes;
        return true;
    };
    auto copy_fp4 = [&](Linear& l) -> bool {
        if (!q8(l) || !kernels::nvfp4p::fits(l.rows, l.cols)) return copy(l);
        void* p = nullptr;
        void* sc = nullptr;
        if (cudaMalloc(&p, kernels::nvfp4p::bytes(l.rows, l.cols)) != cudaSuccess ||
            cudaMalloc(&sc, 16) != cudaSuccess) {
            if (p) cudaFree(p);
            if (sc) cudaFree(sc);
            return false;
        }
        m.owned.push_back(p);
        m.owned.push_back(sc);
        const void* w = l.w;
        const int rows = l.rows, cols = l.cols;
        steps.push_back([=](void*) {
            kernels::nvfp4p::from_q8(w, rows, cols, p, (float*) sc, nullptr);
            return true;
        });
        l.w = p;
        l.scale = (const float*) sc;
        l.type = kTypeNvfp4Planes;
        return true;
    };
    bool ok = true;
    for (BlockDrafter::Layer& D : bd.layers) {
        for (Linear* l : {&D.gate, &D.up, &D.down, &D.q, &D.o}) ok = ok && to_fp4(*l);
        for (Linear* l : {&D.conv_a, &D.conv_f}) ok = ok && in_place(*l);
        if (kernels::nvfp4p::fits(D.k.rows, D.k.cols) && kernels::nvfp4p::fits(D.v.rows, D.v.cols)) {
            // k and v as NVFP4 copies (prompt chunks read their blocks), so q, k and v stay one call
            for (Linear* l : {&D.k, &D.v}) ok = ok && copy_fp4(*l);
        } else {
            ok = ok && copy(D.k) && copy(D.v);
        }
    }
    // fc as NVFP4 copies (prompt chunks still read Q8_0 blocks): halves the bytes read on every target verify window
    for (Linear& l : bd.fc) ok = ok && copy_fp4(l);
    // optional NVFP4 copy of the FP8 LM head's leading kDraftVocab rows for the drafter pass alone (the target model
    // always verifies against the exact FP8 m.w.head); accounts for sessions, lazy prompt scratch and --vram-reserve-mib
    if (ok && m.w.head.type == kTypeF8) {
        const int64_t nv = m.w.head.rows > kDraftVocab ? kDraftVocab : m.w.head.rows;
        if (kernels::nvfp4p::fits(nv, m.w.head.cols)) {
            const size_t hb = kernels::nvfp4p::bytes(nv, m.w.head.cols);
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const uint64_t sess_b =
                (uint64_t) std::max(m.o.sessions, 1) *
                (session_bytes(m.g, m.o.max_context, false, m.o.kv_int8, m.o.kv_resident, m.lb, m.le, m.o.spec < 2) +
                 block_session_bytes(m.g, m.w.dflash));
            const uint64_t chunk_b = plan_chunk(m).total_bytes;
            const uint64_t reserve_b = (uint64_t) std::max(m.o.vram_reserve_mib, 0) << 20;
            if (free_b > sess_b + chunk_b + reserve_b + hb + (512ull << 20)) {
                void* p = nullptr;
                void* sc = nullptr;
                if (cudaMalloc(&p, hb) == cudaSuccess && cudaMalloc(&sc, 16) == cudaSuccess) {
                    m.owned.push_back(p);
                    m.owned.push_back(sc);
                    const void* hw = m.w.head.w;
                    const float* hsc = m.w.head.scale;
                    const float* hrsc = m.w.head.row_scales;
                    const int64_t cols = m.w.head.cols;
                    steps.push_back([=](void*) {
                        kernels::nvfp4p::from_f8(hw, hsc, hrsc, nv, cols, p, (float*) sc, nullptr);
                        return true;
                    });
                    bd.draft_head = {kTypeNvfp4Planes, p, nv, cols, (const float*) sc, nullptr};
                    basalt::log::info("basalt dense", "block drafter head: NVFP4 copy of %lld rows (%.0f MiB)",
                                      (long long) nv, (double) hb / 1048576.0);
                } else {
                    if (p) cudaFree(p);
                    if (sc) cudaFree(sc);
                    (void) cudaGetLastError();
                    basalt::log::info("basalt dense", "block drafter head: using FP8 head (NVFP4 copy allocation failed)");
                }
            } else {
                basalt::log::info("basalt dense", "block drafter head: using FP8 head (tight VRAM headroom)");
            }
        }
    }
    if (!ok) {
        err = m.o.arch + ": laying out the block drafter for decode windows failed";
        return false;
    }
    // the workspace for the drafter's calls (each matrix, q + k + v, k + v, gate + up)
    namespace wg = kernels::wgemm;
    size_t need_p = m.ws.w8.partial_floats, need_t = m.ws.w8.tickets_n;
    auto need = [&](std::initializer_list<const Linear*> ls, wg::Epilogue epi = wg::Epilogue::Store) {
        wg::Call c;
        c.fmt = wg::Format::Q8;
        c.n = 0;
        for (const Linear* l : ls) {
            if (l->type != kTypeQ8P && l->type != kTypeNvfp4Planes) return;
            c.fmt = l->type == kTypeQ8P ? wg::Format::Q8 : wg::Format::Nvfp4;
            c.K = l->cols;
            c.m[c.n++].rows = l->rows;
        }
        if (epi == wg::Epilogue::SwiGLU && c.fmt == wg::Format::Nvfp4) c.epi = wg::Epilogue::SwiGLU;
        size_t p, t;
        wg::workspace_need(c, p, t);
        need_p = std::max(need_p, p);
        need_t = std::max(need_t, t);
    };
    for (const BlockDrafter::Layer& D : bd.layers) {
        for (const Linear* l : {&D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down, &D.conv_a, &D.conv_f}) need({l});
        need({&D.q, &D.k, &D.v});
        need({&D.k, &D.v});
        need({&D.gate, &D.up}, wg::Epilogue::SwiGLU);
    }
    for (const Linear& l : bd.fc) need({&l});
    if (bd.draft_head.w) need({&bd.draft_head});
    if (need_p > m.ws.w8.partial_floats || need_t > m.ws.w8.tickets_n) {
        void *p = nullptr, *t = nullptr;
        if (cudaMalloc(&p, std::max<size_t>(need_p, 1) * 4) != cudaSuccess || cudaMalloc(&t, need_t * 4) != cudaSuccess ||
            cudaMemset(t, 0, need_t * 4) != cudaSuccess) {
            err = m.o.arch + ": out of device memory for the drafter's workspace";
            return false;
        }
        m.owned.push_back(p);
        m.owned.push_back(t);
        m.ws.w8.partial = (float*) p;
        m.ws.w8.tickets = (int32_t*) t;
        m.ws.w8.partial_floats = need_p;
        m.ws.w8.tickets_n = need_t;
    }
    return true;
}

}  // namespace

bool attach_drafter(Model& m, std::string& err) {
    if (m.drafter_steps.empty()) return true;
    void* tmp = nullptr;
    bool ok = m.drafter_tmp == 0 || cudaMalloc(&tmp, m.drafter_tmp) == cudaSuccess;
    for (const auto& step : m.drafter_steps) ok = ok && step(tmp);
    if (tmp) cudaFree(tmp);
    m.drafter_steps.clear();
    if (!ok || cudaDeviceSynchronize() != cudaSuccess) {
        err = m.o.arch + ": laying out the block drafter for decode windows failed";
        return false;
    }
    return true;
}

namespace {
// a cell of one attention layer, K and V: fp16, or int8 codes with their fp16 scales
uint64_t cell_bytes(const Geometry& g, bool kv_int8) {
    return kv_int8 ? (uint64_t) g.n_kv * (g.head_dim + g.head_dim / 64 * 2) * 2 : (uint64_t) g.n_kv * g.head_dim * 2 * 2;
}
}  // namespace

int64_t resident_cells(int64_t max_context, int64_t kv_resident) {
    return kv_resident > 0 && kv_resident < max_context ? kv_resident : max_context;
}

uint64_t shared_bytes(const Geometry& g, int64_t max_context, bool kv_int8, int64_t kv_resident) {
    const uint64_t stage = g.attn_layers() > 0 ? (uint64_t) (max_context - resident_cells(max_context, kv_resident)) * cell_bytes(g, kv_int8) : 0;
    return window_bytes(g, BlockDrafter{}, max_context) + chunk_bytes(g) + stage + (512ull << 20);
}

uint64_t session_host_bytes(const Geometry& g, int64_t max_context, bool kv_int8, int64_t kv_resident) {
    return (uint64_t) (max_context - resident_cells(max_context, kv_resident)) * g.attn_layers() * cell_bytes(g, kv_int8);
}

uint64_t session_layer_bytes(const Geometry& g, int il, int64_t max_context, bool kv_int8, int64_t kv_resident,
                             bool gdn_alt) {
    if (g.kinds[(size_t) il] == Mixer::Gdn) {
        const uint64_t C = (uint64_t) g.conv_channels();
        return (gdn_alt ? 2ull : 1ull) * ((uint64_t) g.gdn_state * g.gdn_v_heads * g.gdn_state + C * 3) * 4;
    }
    return (uint64_t) resident_cells(max_context, kv_resident) * cell_bytes(g, kv_int8) + 4 * 256;
}

uint64_t session_bytes(const Geometry& g, int64_t max_context, bool mtp, bool kv_int8, int64_t kv_resident, int lb, int le,
                       bool gdn_alt) {
    uint64_t b = 64 * 256;
    for (int il = lb; il < (le > 0 ? le : g.n_layers); ++il)
        b += session_layer_bytes(g, il, max_context, kv_int8, kv_resident, gdn_alt);
    if (lb > 0) return b;
    const uint64_t drafter = mtp ? (uint64_t) max_context * cell_bytes(g, false) + ((uint64_t) (kWindow + 3) * g.n_embd + g.n_vocab + (uint64_t) kWindow * 64) * 4
                                 : 0;
    return b + drafter + (uint64_t) kWindow * 4;
}

uint64_t block_session_bytes(const Geometry& g, const BlockDrafter& bd) {
    if (!bd.present) return 0;
    return bd.layers.size() * 2 * (uint64_t) (bd.window + 256) * bd.n_kv * bd.head_dim * 2 + (uint64_t) kWindow * (g.n_embd * 4 + 16) +
           64 * 256;
}

SplitInputs split_inputs(const Geometry& g, const Options& o, const BlockDrafter& bd, const std::vector<uint64_t>& layer_weights,
                         uint64_t main_weights, SplitCard main_card, SplitCard peer_card) {
    SplitInputs in;
    const bool mtp = o.drafter == Drafter::Mtp;
    in.layer = layer_weights;
    in.slots = std::max(o.sessions, 1);
    uint64_t layers = 0;
    const bool gdn_alt = o.spec < 2;
    for (int il = 0; il < g.n_layers; ++il) {
        in.per_slot.push_back(session_layer_bytes(g, il, o.max_context, o.kv_int8, o.kv_resident, gdn_alt));
        layers += in.per_slot.back();
    }
    const uint64_t slot_rest = session_bytes(g, o.max_context, mtp, o.kv_int8, o.kv_resident, 0, 0, gdn_alt) - layers +
                               (o.drafter == Drafter::Block ? block_session_bytes(g, bd) : 0);
    const uint64_t scratch = shared_bytes(g, o.max_context, o.kv_int8, o.kv_resident);
    in.main_fixed = main_weights + scratch + (uint64_t) in.slots * slot_rest + stage_main_bytes(g, bd);
    in.peer_fixed = scratch + 2ull * kChunk * g.n_embd * 2;   // its own scratch and the landing residuals
    in.main = main_card;
    in.peer = peer_card;
    return in;
}

Model::~Model() {
    for (BatchGraph& g : batch_graphs)
        if (g.exec) cudaGraphExecDestroy(g.exec);
    for (BatchGraph& g : draft_graphs)
        if (g.exec) cudaGraphExecDestroy(g.exec);
    for (BatchGraph& g : ctx_graphs)
        if (g.exec) cudaGraphExecDestroy(g.exec);
    if (gate_side) cudaStreamDestroy(gate_side);
    if (gate_fork) cudaEventDestroy(gate_fork);
    if (gate_join) cudaEventDestroy(gate_join);
    if (stage_side) cudaStreamDestroy(stage_side);
    if (stage_free) cudaEventDestroy(stage_free);
    if (stage_ready) cudaEventDestroy(stage_ready);
    for (void* p : owned) cudaFree(p);
    vision_detach();
}

bool Model::vision_attach(int tokens, std::string& err) {
    if (!encoder) {
        err = o.arch + ": started without --vision";
        return false;
    }
    const size_t need = encoder->arena_bytes(tokens);
    if (enc_arena && enc_bytes >= need) return true;
    vision_detach();
    if (cudaMalloc(&enc_arena, need) != cudaSuccess) {
        cudaGetLastError();
        enc_arena = nullptr;
        err = o.arch + ": no device memory for the image encoder (" + std::to_string((unsigned long long) (need >> 20)) + " MiB)";
        return false;
    }
    enc_bytes = need;
    if (!encoder->attach(enc_arena, enc_bytes, tokens, err)) {
        vision_detach();
        return false;
    }
    return true;
}

void Model::vision_detach() {
    if (!enc_arena) return;
    if (encoder) encoder->detach();
    cudaFree(enc_arena);
    enc_arena = nullptr;
    enc_bytes = 0;
}

bool Model::ready_products(std::string& err) {
    mmq = std::make_unique<prefill::mmq::Context>();
    std::vector<const Layer*> all;
    for (const Layer& L : w.layers) all.push_back(&L);
    if (o.drafter == Drafter::Mtp) all.push_back(&w.mtp.layer);
    std::vector<const Linear*> ls = {&w.head};
    if (o.drafter == Drafter::Mtp) ls.push_back(&w.mtp.fc);
    if (o.drafter == Drafter::Block) {
        for (const Linear& l : w.dflash.fc) ls.push_back(&l);
        for (const BlockDrafter::Layer& D : w.dflash.layers)
            for (const Linear* l : {&D.q, &D.k, &D.v, &D.o, &D.gate, &D.up, &D.down, &D.conv_a, &D.conv_f}) ls.push_back(l);
    }
    for (const Layer* Lp : all)
        for (const Linear* l : {&Lp->qkv, &Lp->z, &Lp->out, &Lp->q, &Lp->k, &Lp->v, &Lp->o, &Lp->gate, &Lp->up, &Lp->down})
            if (l->w) ls.push_back(l);
    for (const Linear* l : ls) {
        if (l->type == kTypeF8 || l->type == kTypeNvfp4Planes) continue;   // the window gemm's, not MMQ's
        if (!prefill::mmq::fits(l->type, l->rows)) {
            err = o.arch + ": the products have no tile for a " + std::to_string(l->rows) + "-row matrix";
            return false;
        }
        prefill::mmq::load(l->type, l->rows);
    }
    return true;
}

bool Model::ensure_chunk(std::string& err) {
    if (chunk_ready) return true;
    const Geometry& gg = g;
    const ChunkPlan cp = plan_chunk(*this);
    if (!pre_arena.reserve(cp.total_bytes)) {
        err = o.arch + ": out of device memory for the prompt workspaces";
        return false;
    }
    const size_t T = kChunk, H = (size_t) gg.n_embd, AQ = (size_t) gg.n_head * gg.head_dim, C = (size_t) gg.conv_channels();
    const size_t VD = (size_t) gg.value_width(), HV = (size_t) gg.gdn_v_heads, KV = (size_t) gg.n_kv * gg.head_dim;
    ChunkScratch& b = cs;
    b = {};
    b.x = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * H));   // BF16 residual stream (residual(b))
    b.xn = pre_arena.take<float>(T * H);
    b.tmp = pre_arena.take<float>(T * H);
    b.xq = cp.xq_bytes ? pre_arena.take_bytes(cp.xq_bytes) : nullptr;
    b.tok = pre_arena.take<int32_t>(T);
    b.pos = pre_arena.take<int32_t>(3 * T);
    b.iota = pre_arena.take<int32_t>(T);
    b.bounds = pre_arena.take<int32_t>(2);
    if (cp.mtp) {
        b.cat = pre_arena.take<float>(T * H);
        b.hn = pre_arena.take<float>((T + 1) * H);
    }
    if (cp.block) b.dacc = pre_arena.take<float>(T * H);
    // the FP4 activations (fp4_dense) of the widest product input and their token scales
    b.a4 = cp.a4_bytes ? pre_arena.take_bytes(cp.a4_bytes) : nullptr;
    b.gs = pre_arena.take<float>(T);
    b.hmax = pre_arena.take<float>(T);
    const size_t phase = pre_arena.mark();
    if (gg.gdn_layers()) {
        if (cp.gdn_bf16) {
            b.qkv = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * C));
            b.hc = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * C));
            b.z = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * VD));
            b.y = pre_arena.take<float>(T * VD);
            b.yn = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * VD));
            b.gate = pre_arena.take<float>(T * HV);
            b.beta = pre_arena.take<float>(T * HV);
        } else {
            b.qkv = pre_arena.take<float>(T * C);
            b.hc = pre_arena.take<float>(T * C);
            b.z = pre_arena.take<float>(T * VD);
            b.y = pre_arena.take<float>(T * VD);
            b.yn = pre_arena.take<float>(T * VD);
            b.gate = pre_arena.take<float>(T * HV);
            b.beta = pre_arena.take<float>(T * HV);
            b.ab = pre_arena.take<float>(T * 2 * HV);
        }
        b.work = pre_arena.take_bytes(kernels::gdn_work_bytes(T));
        if (!b.work) {
            err = o.arch + ": the prompt GDN workspaces do not fit their arena";
            return false;
        }
        pre_arena.rewind(phase);
    }
    if (gg.attn_layers() || cp.mtp) {
        if (cp.attn_bf16) {
            b.qfull = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * 2 * AQ));
            b.k = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * KV));
            b.v = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * KV));
            b.qo = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * AQ));
            b.ao = cp.kv_int8 ? reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * AQ)) : pre_arena.take<float>(T * AQ);
            b.ag = cp.kv_int8 ? reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * AQ)) : pre_arena.take<float>(T * AQ);
        } else {
            b.qfull = pre_arena.take<float>(T * 2 * AQ);
            b.k = pre_arena.take<float>(T * KV);
            b.v = pre_arena.take<float>(T * KV);
            b.qo = pre_arena.take<float>(T * AQ);
            b.ao = pre_arena.take<float>(T * AQ);
            b.ag = pre_arena.take<float>(T * AQ);
        }
        if (cp.kv_int8)
            b.attn_split = pre_arena.take<float>(kernels::gqa_prompt_split_floats((int64_t) T, gg.n_head, kAttnSplits));
        if (!b.ag || (cp.kv_int8 && !b.attn_split)) {
            err = o.arch + ": the prompt attention workspaces do not fit their arena";
            return false;
        }
        pre_arena.rewind(phase);
    }
    if (cp.ffn_fp4) {
        b.h = reinterpret_cast<float*>(pre_arena.take<uint16_t>(T * (size_t) gg.n_ff));
    } else {
        b.g = pre_arena.take<float>(T * (size_t) gg.n_ff);
        b.u = pre_arena.take<float>(T * (size_t) gg.n_ff);
        b.h = pre_arena.take<float>(T * (size_t) gg.n_ff);
    }
    if (cp.block) {
        pre_arena.rewind(phase);
        const size_t BKV = (size_t) w.dflash.n_kv * (size_t) w.dflash.head_dim;
        b.dk = pre_arena.take<float>(T * BKV);
        b.dv = pre_arena.take<float>(T * BKV);
    }
    if (!b.h || !b.x || (cp.xq_bytes && !b.xq) || (cp.a4_bytes && !b.a4) || (cp.block && (!b.dacc || !b.dk || !b.dv))) {
        err = o.arch + ": the prompt workspaces do not fit their arena";
        return false;
    }
    prefill::mmq::iota(b.iota, (int64_t) T, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = o.arch + ": preparing the prompt kernels failed";
        return false;
    }
    chunk_ready = true;
    return true;
}

void Model::release_chunk() {
    if (!chunk_ready) return;
    pre_arena.reserve(0);
    cs = {};
    chunk_ready = false;
}

ModelPtr make_model(const Geometry& g, const Weights& w, const Options& o, std::string& err) {
    ModelPtr m(new Model());
    m->g = g;
    m->w = w;
    m->o = o;
    if (const std::string e = g.check(); !e.empty()) {
        err = o.arch + ": " + e;
        return nullptr;
    }
    m->lb = o.layer_begin;
    m->le = o.layer_end > 0 ? o.layer_end : g.n_layers;
    if (m->lb < 0 || m->lb >= m->le || m->le > g.n_layers) {
        err = o.arch + ": layers " + std::to_string(m->lb) + "-" + std::to_string(m->le - 1) + " of " + std::to_string(g.n_layers);
        return nullptr;
    }
    for (int il = 0; il < g.n_layers; ++il) {
        const bool here = il >= m->lb && il < m->le;
        const Mixer k = g.kinds[(size_t) il];
        m->gdn_index.push_back(here && k == Mixer::Gdn ? m->n_gdn++ : -1);
        m->attn_index.push_back(here && k == Mixer::Attention ? m->n_attn++ : -1);
    }
    if (o.vision) {   // the projector stays in page-locked RAM; each request's pictures take it to the card
        m->mmproj = std::make_unique<vision::Mmproj>();
        m->encoder = std::make_unique<vision::CudaEncoder>();
        if (o.mmproj.empty()) {
            err = o.arch + ": --vision needs the pack's vision/mmproj section";
            return nullptr;
        }
        if (!m->mmproj->open(o.mmproj, err) || !m->encoder->stage(*m->mmproj, true, err) || !m->encoder->create(err)) {
            err = o.arch + ": vision: " + err;
            return nullptr;
        }
        if (m->mmproj->hp().proj != g.n_embd) {
            err = o.arch + ": the vision projector's width differs from the model's";
            return nullptr;
        }
        m->mmproj->release();
        basalt::log::info("basalt dense", "image encoder: %.2f GiB staged in page-locked RAM, on the card per request",
                          (double) m->encoder->weights_bytes() / 1073741824.0);
    }
    const int rec_rows = o.sessions > 1 ? kBatch : kWindow;
    if (!m->dec_arena.reserve(window_bytes(g, o.drafter == Drafter::Block ? w.dflash : BlockDrafter{}, o.max_context, m->n_gdn, rec_rows))) {
        err = o.arch + ": out of device memory for the decode scratch";
        return nullptr;
    }
    const size_t W = kBatch, H = (size_t) g.n_embd, AQ = (size_t) g.n_head * g.head_dim, KV = (size_t) g.n_kv * g.head_dim;
    WindowScratch& b = m->ws;
    Arena& a = m->dec_arena;
    b.x = a.take<uint16_t>(W * H);
    b.xn = a.take<float>(W * H);
    b.tmp = a.take<float>(W * H);
    b.dacc = a.take<float>(W * H);
    b.xq = a.take_bytes(act_slot(g));
    b.hq = a.take_bytes(act_slot(g));
    b.x8 = a.take_bytes(q8_slot(g));
    wgemm_need(g, b.w8.partial_floats, b.w8.tickets_n);
    b.w8.partial = a.take<float>(std::max<size_t>(b.w8.partial_floats, 1));
    b.w8.tickets = a.take<int32_t>(std::max<size_t>(b.w8.tickets_n, 1));   // zero: the arena is cleared below
    b.z = a.take<float>(W * g.value_width());
    b.qfull = a.take<float>(W * 2 * AQ);
    b.k = a.take<float>(W * KV);
    b.v = a.take<float>(W * KV);
    b.qo = a.take<float>(W * AQ);
    b.ao = a.take<float>(W * AQ);
    b.ag = a.take<float>(W * AQ);
    b.g = a.take<float>(W * g.n_ff);
    b.u = a.take<float>(W * g.n_ff);
    b.h = a.take<float>(W * g.n_ff);
    if (g.attn_layers() > 0) b.attn = a.take<float>(kernels::gqa_decode_scratch_floats(o.max_context, kBatch, g.n_head));
    b.tok = a.take<int32_t>(W);
    b.pos = a.take<int32_t>(W * 3);
    b.cat = a.take<float>(W * 2 * H);
    b.lp_scratch = a.take<float>(kernels::row_logprobs_scratch_floats(kBatch, g.n_vocab));
    if (o.drafter == Drafter::Block) {
        const BlockDrafter& bd = w.dflash;
        const size_t AQd = (size_t) bd.n_head * bd.head_dim, KVd = (size_t) bd.n_kv * bd.head_dim;
        b.bq = a.take<float>(W * AQd);
        b.bk = a.take<float>(W * KVd);
        b.bv = a.take<float>(W * KVd);
        b.bqo = a.take<float>(W * AQd);
        b.bao = a.take<float>(W * AQd);
        b.dyn = a.take<float>(W * bd.layers[0].conv_a.rows);
        b.bh = a.take<float>(W * H);
        b.blp = a.take<float>(W * kernels::kLogprobStride);
        b.bsel = a.take<float>(W * bd.rank);
        b.bpos = a.take<int32_t>(W * 3);
        b.ctx_pos0 = a.take<int32_t>(W);
        b.dwin = a.take<DraftWin>(W);
        b.battn = a.take<float>(kernels::gqa_decode_scratch_floats(bd.window + 256, kBatch, bd.n_head, bd.head_dim));
    }
    b.y = a.take<float>(W * g.value_width());
    b.hl = a.take<float>(W * (size_t) g.n_vocab);
    b.iota = a.take<int32_t>(W);
    b.bounds = a.take<int32_t>(2);
    b.pos0 = a.take<int32_t>(2);
    if (m->n_gdn > 0) {
        const size_t ng = (size_t) m->n_gdn, rr = (size_t) rec_rows;
        const size_t C = (size_t) g.conv_channels(), HKV = C - (size_t) g.gdn_k_heads * (size_t) g.gdn_state;
        const size_t HV = (size_t) g.gdn_v_heads;
        b.rec_rows = rec_rows;
        b.rec_qkv = a.take<float>(ng * rr * C);
        b.rec_h = a.take<float>(ng * rr * HKV);
        b.rec_gate = a.take<float>(ng * rr * HV);
        b.rec_beta = a.take<float>(ng * rr * HV);
        b.gab = a.take<float>(2 * W * HV);
        if (cudaStreamCreateWithFlags(&m->gate_side, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&m->gate_fork, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&m->gate_join, cudaEventDisableTiming) != cudaSuccess) {
            err = o.arch + ": creating the GDN gate stream failed";
            return nullptr;
        }
    }
    b.btab = a.take<BatchTable>(1);
    if (!b.btab) {
        err = o.arch + ": the decode scratch does not fit its arena";
        return nullptr;
    }
    // rows past a window's last stay finite: they are quantized with it
    if (cudaMemset(b.x, 0, a.mark()) != cudaSuccess) {   // b.x is the arena's first buffer
        err = o.arch + ": clearing the decode scratch failed";
        return nullptr;
    }
    prefill::mmq::iota(b.iota, kBatch, nullptr);
    if (const int64_t kept = resident_cells(o.max_context, o.kv_resident); kept < o.max_context && g.attn_layers() > 0) {
        // the prompt chunks' VRAM copy of one layer's far cells, laid out as a session's far buffers
        const size_t cells = (size_t) (o.max_context - kept) * g.n_kv * g.head_dim;
        void* p = nullptr;
        if (cudaMalloc(&p, (size_t) (o.max_context - kept) * cell_bytes(g, o.kv_int8)) != cudaSuccess ||
            cudaStreamCreateWithFlags(&m->stage_side, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&m->stage_free, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&m->stage_ready, cudaEventDisableTiming) != cudaSuccess) {
            if (p) cudaFree(p);
            err = o.arch + ": out of device memory for the streamed K/V's prompt copy";
            return nullptr;
        }
        m->owned.push_back(p);
        kernels::KvFar& f = m->stage;
        f.at = kept;
        if (o.kv_int8) {
            f.q8.k = (int8_t*) p;
            f.q8.v = f.q8.k + cells;
            f.q8.ks = (uint16_t*) (f.q8.v + cells);
            f.q8.vs = f.q8.ks + cells / 64;
        } else {
            f.k = (uint16_t*) p;
            f.v = f.k + cells;
        }
    }
    if (!decode_drafter(*m, err)) return nullptr;
    if (!m->ready_products(err) || cudaDeviceSynchronize() != cudaSuccess) {
        if (err.empty()) err = o.arch + ": readying the products failed";
        return nullptr;
    }
    return m;
}

void destroy(Model* m) { delete m; }

}  // namespace basalt::dense
