// SPDX-FileCopyrightText: 2026 The MoonMath.ai
// SPDX-FileCopyrightText: 2026 Luma AI, Inc.
// SPDX-License-Identifier: MIT
// Pybind11 API for online_flash_4_amd — ROCm CDNA3 (MI300X/gfx942) fused forward attention,
// with a RingAttention online (chunked-KV, HBM-threaded state) path. Adapted from the
// bindings in MoonMath's amd-kernels (csrc/attention_api.cpp).

#include "sparse_args.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// Extern "C" launchers from the three round-mode TUs
extern "C" {
int launch_v_transpose_rtna(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream);
int launch_attention_forward_rtna(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int kv_heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    int layout,
    void* workspace,
    void* stream);

int launch_v_transpose_rtne(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream);
int launch_attention_forward_rtne(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int kv_heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    int layout,
    void* workspace,
    void* stream);

int launch_v_transpose_rtz(
    const void* v, void* v_t, int n_rows, int seq_len_per_head, int heads, int layout, void* stream);
int launch_attention_forward_rtz(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int kv_heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    int layout,
    void* workspace,
    void* stream);

// LiteAttention launchers (one per round-mode TU). read_list/write_list/must_do_list are int16.
int launch_attention_forward_lite_rtna(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int kv_heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    const void* read_list,
    void* write_list,
    const void* must_do_list,
    float threshold,
    int phase,
    int layout,
    void* workspace,
    void* stream);
int launch_attention_forward_lite_rtne(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int kv_heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    const void* read_list,
    void* write_list,
    const void* must_do_list,
    float threshold,
    int phase,
    int layout,
    void* workspace,
    void* stream);
int launch_attention_forward_lite_rtz(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int batch,
    int heads,
    int kv_heads,
    int seq_len,
    int seq_len_kv,
    int head_dim,
    const void* read_list,
    void* write_list,
    const void* must_do_list,
    float threshold,
    int phase,
    int layout,
    void* workspace,
    void* stream);

// Dense tail KV-split workspace size (round-mode-independent; one per TU).
size_t tail_workspace_bytes_rtna(void);
size_t tail_workspace_bytes_rtne(void);
size_t tail_workspace_bytes_rtz(void);

// Online (chunked / ring) attention: per-chunk partial + LSE merge (one set per round-mode TU).
// Online (RingAttention): (q, k, v_t, out, B, H, Sq, Skv_chunk, D, layout, ring_state, is_first, is_last, stream)
int launch_attention_online_rtna(const void*, const void*, const void*, void*, int, int, int, int, int, int, int, void*, int, int, void*, const SparseArgs*);
int launch_attention_online_rtne(const void*, const void*, const void*, void*, int, int, int, int, int, int, int, void*, int, int, void*, const SparseArgs*);
int launch_attention_online_rtz(const void*, const void*, const void*, void*, int, int, int, int, int, int, int, void*, int, int, void*, const SparseArgs*);
size_t ring_state_bytes_rtna(int, int, int);
size_t ring_state_bytes_rtne(int, int, int);
size_t ring_state_bytes_rtz(int, int, int);

// Lite LPT order-buffer size (one int per work-group). One per round-mode TU.
size_t lite_workspace_bytes_rtna(int batch, int heads, int seq_len);
size_t lite_workspace_bytes_rtne(int batch, int heads, int seq_len);
size_t lite_workspace_bytes_rtz(int batch, int heads, int seq_len);
}

// Helper: validate tensor properties
static void check_tensor(const at::Tensor& t, const std::string& name, bool check_contiguous = true) {
  if (!t.defined()) {
    throw std::invalid_argument(name + " must be a valid tensor");
  }
  if (t.scalar_type() != at::kBFloat16) {
    throw std::invalid_argument(name + " must be bfloat16, got " + std::string(at::toString(t.scalar_type())));
  }
  if (t.dim() != 4) {
    throw std::invalid_argument(name + " must be 4-D, got " + std::to_string(t.dim()) + "-D");
  }
  if (check_contiguous && !t.is_contiguous()) {
    throw std::invalid_argument(name + " must be contiguous");
  }
  if (!t.device().is_cuda()) {
    throw std::invalid_argument(name + " must be on a CUDA/HIP device (GPU-only), got device=" + t.device().str());
  }
}

at::Tensor forward(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    std::optional<at::Tensor> out_,
    std::string round_mode,
    std::string layout) {
  // Validate inputs
  check_tensor(q, "q");
  check_tensor(k, "k");
  check_tensor(v, "v");

  // Validate round_mode
  if (round_mode != "rtna" && round_mode != "rtne" && round_mode != "rtz") {
    throw std::invalid_argument("round_mode must be 'rtna', 'rtne', or 'rtz', got '" + round_mode + "'");
  }

  // Validate layout
  bool bshd = (layout == "bshd");
  if (!bshd && layout != "bhsd") {
    throw std::invalid_argument("layout must be 'bhsd' or 'bshd', got '" + layout + "'");
  }
  int layout_int = bshd ? 1 : 0;

  // Extract dimensions (layout-aware)
  int sax = bshd ? 1 : 2;  // seq axis
  int hax = bshd ? 2 : 1;  // heads axis

  int B = q.size(0);
  int Hq = q.size(hax);
  int Sq = q.size(sax);
  int D = q.size(3);
  int Skv = k.size(sax);
  int Hkv = k.size(hax);

  // Validate shapes
  if (D != 128) {
    throw std::invalid_argument("head_dim must be 128, got " + std::to_string(D));
  }

  if (k.size(0) != B || k.size(3) != D) {
    std::ostringstream oss;
    oss << "k shape mismatch: expected batch=" << B << ", head_dim=" << D << ", got k.shape=" << k.sizes();
    throw std::invalid_argument(oss.str());
  }
  if (Hq <= 0 || Hkv <= 0 || Hq < Hkv || Hq % Hkv != 0) {
    throw std::invalid_argument(
        "GQA head mismatch: require num_q_heads >= num_kv_heads and num_q_heads % num_kv_heads == 0, got "
        "num_q_heads=" + std::to_string(Hq) + ", num_kv_heads=" + std::to_string(Hkv));
  }

  if (v.size(0) != B || v.size(hax) != Hkv || v.size(sax) != Skv || v.size(3) != D) {
    std::ostringstream oss;
    oss << "v shape mismatch: must match k.shape, got v.shape=" << v.sizes() << ", k.shape=" << k.sizes();
    throw std::invalid_argument(oss.str());
  }

  if (Sq <= 0 || Skv <= 0) {
    throw std::invalid_argument(
        "seq_len must be positive, got Sq=" + std::to_string(Sq) + ", Skv=" + std::to_string(Skv));
  }

  // Check devices match
  if (q.device() != k.device() || q.device() != v.device()) {
    throw std::invalid_argument("q, k, v must be on the same device");
  }

  // Validate or allocate output
  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    check_tensor(out, "out");
    if (out.sizes() != q.sizes()) {
      std::ostringstream oss;
      oss << "out shape must match q.shape, got out.shape=" << out.sizes() << ", q.shape=" << q.sizes();
      throw std::invalid_argument(oss.str());
    }
    if (out.device() != q.device()) {
      throw std::invalid_argument("out must be on the same device as q");
    }
  } else {
    out = at::empty_like(q);
  }

  // Set device guard and get stream
  const c10::cuda::CUDAGuard device_guard(q.device());
  void* stream = (void*)at::cuda::getCurrentCUDAStream(q.device().index()).stream();

  // Allocate v_t (per-head padded to 64-row boundary)
  int Skv_pad = ((Skv + 63) / 64) * 64;
  at::Tensor v_t = at::empty({B, Hkv, Skv_pad, D}, v.options());

  // Dispatch on round_mode
  int (*vt_fn)(const void*, void*, int, int, int, int, void*);
  int (*fwd_fn)(const void*, const void*, const void*, void*, int, int, int, int, int, int, int, void*, void*);
  size_t (*wsbytes_fn)(void);

  if (round_mode == "rtna") {
    vt_fn = launch_v_transpose_rtna;
    fwd_fn = launch_attention_forward_rtna;
    wsbytes_fn = tail_workspace_bytes_rtna;
  } else if (round_mode == "rtne") {
    vt_fn = launch_v_transpose_rtne;
    fwd_fn = launch_attention_forward_rtne;
    wsbytes_fn = tail_workspace_bytes_rtne;
  } else {  // rtz
    vt_fn = launch_v_transpose_rtz;
    fwd_fn = launch_attention_forward_rtz;
    wsbytes_fn = tail_workspace_bytes_rtz;
  }

  // Tail KV-split workspace: recovers the stranded fractional round (dense). Allocated via the
  // caching allocator, so it's effectively reused across calls. Dev toggle OF4AMD_NO_TAIL=1
  // passes a null workspace (plan_tail_split disables the split) for A/B against the same build.
  const bool no_tail = (getenv("OF4AMD_NO_TAIL") != nullptr);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (!no_tail) {
    workspace = at::empty({(int64_t)wsbytes_fn()}, at::TensorOptions().dtype(at::kByte).device(q.device()));
    workspace_ptr = workspace.data_ptr();
  }

  // Launch V transpose
  int rc = vt_fn(v.data_ptr(), v_t.data_ptr(), B * Hkv * Skv, Skv, Hkv, layout_int, stream);
  if (rc != 0) {
    throw std::runtime_error("launch_v_transpose returned error code " + std::to_string(rc));
  }

  // Launch attention forward (with the tail workspace)
  rc = fwd_fn(
      q.data_ptr(), k.data_ptr(), v_t.data_ptr(), out.data_ptr(), B, Hq, Hkv, Sq, Skv, D, layout_int, workspace_ptr, stream);
  if (rc != 0) {
    throw std::runtime_error("launch_attention_forward returned error code " + std::to_string(rc));
  }

  return out;
}

// Online (chunked / ring) attention: split K/V into `num_chunks` contiguous KV chunks, run each chunk as a SEPARATE
// kernel launch producing an independent fp32 (o,m,l) partial, then one online-softmax LSE merge → bf16 Out.
// Numerically identical to full attention (num_chunks==1 degenerates to forward()). Demonstrates KV streaming on the
// fast forward inner loop. Same rounding/layout options as forward().
namespace {
// Common online setup: validate + resolve layout/dims + round-mode function pointers.
struct OnlineCtx {
  bool bshd;
  int layout_int, sax, hax, B, Hq, Sq, D;
  int (*vt_fn)(const void*, void*, int, int, int, int, void*);
  int (*online_fn)(const void*, const void*, const void*, void*, int, int, int, int, int, int, int, void*, int, int, void*, const SparseArgs*);
  size_t (*ring_bytes_fn)(int, int, int);
};
static OnlineCtx online_ctx(const at::Tensor& q, const std::string& round_mode, const std::string& layout) {
  if (round_mode != "rtna" && round_mode != "rtne" && round_mode != "rtz")
    throw std::invalid_argument("round_mode must be 'rtna', 'rtne', or 'rtz', got '" + round_mode + "'");
  bool bshd = (layout == "bshd");
  if (!bshd && layout != "bhsd")
    throw std::invalid_argument("layout must be 'bhsd' or 'bshd', got '" + layout + "'");
  OnlineCtx c;
  c.bshd = bshd; c.layout_int = bshd ? 1 : 0; c.sax = bshd ? 1 : 2; c.hax = bshd ? 2 : 1;
  c.B = q.size(0); c.Hq = q.size(c.hax); c.Sq = q.size(c.sax); c.D = q.size(3);
  if (c.D != 128) throw std::invalid_argument("head_dim must be 128, got " + std::to_string(c.D));
  if (c.Sq <= 0) throw std::invalid_argument("Sq (query seq_len) must be positive, got " + std::to_string(c.Sq));
  if (round_mode == "rtna") {
    c.vt_fn = launch_v_transpose_rtna; c.online_fn = launch_attention_online_rtna; c.ring_bytes_fn = ring_state_bytes_rtna;
  } else if (round_mode == "rtne") {
    c.vt_fn = launch_v_transpose_rtne; c.online_fn = launch_attention_online_rtne; c.ring_bytes_fn = ring_state_bytes_rtne;
  } else {
    c.vt_fn = launch_v_transpose_rtz; c.online_fn = launch_attention_online_rtz; c.ring_bytes_fn = ring_state_bytes_rtz;
  }
  return c;
}

static int validate_online_kv(const OnlineCtx& c, const at::Tensor& k, const at::Tensor& v) {
  const int Hkv = k.size(c.hax);
  if (k.size(0) != c.B || k.size(3) != c.D) {
    std::ostringstream oss;
    oss << "k shape mismatch: expected batch=" << c.B << ", head_dim=" << c.D << ", got k.shape=" << k.sizes();
    throw std::invalid_argument(oss.str());
  }
  if (v.sizes() != k.sizes()) {
    std::ostringstream oss;
    oss << "v shape mismatch: must match k.shape, got v.shape=" << v.sizes() << ", k.shape=" << k.sizes();
    throw std::invalid_argument(oss.str());
  }
  if (c.Hq <= 0 || Hkv <= 0 || c.Hq < Hkv || c.Hq % Hkv != 0) {
    throw std::invalid_argument(
        "GQA head mismatch: require num_q_heads >= num_kv_heads and num_q_heads % num_kv_heads == 0, got "
        "num_q_heads=" + std::to_string(c.Hq) + ", num_kv_heads=" + std::to_string(Hkv));
  }
  return Hkv;
}

static at::Tensor validate_or_allocate_out(const at::Tensor& q, const std::optional<at::Tensor>& out_) {
  if (!out_.has_value()) return at::empty_like(q);
  at::Tensor out = out_.value();
  check_tensor(out, "out");
  if (out.sizes() != q.sizes()) {
    std::ostringstream oss;
    oss << "out shape must match q.shape, got out.shape=" << out.sizes() << ", q.shape=" << q.sizes();
    throw std::invalid_argument(oss.str());
  }
  if (out.device() != q.device()) throw std::invalid_argument("out must be on the same device as q");
  return out;
}

static at::Tensor allocate_ring_state(const OnlineCtx& c, const at::Device& device) {
  const int nqb = (c.Sq + 383) / 384;
  const int64_t total_bytes = (int64_t)c.ring_bytes_fn(c.B, c.Hq, c.Sq);
  const int64_t block_bytes = total_bytes / ((int64_t)c.B * c.Hq * nqb);
  at::Tensor state = at::empty(
      {c.B, c.Hq, nqb, 1, 1, block_bytes}, at::TensorOptions().dtype(at::kByte).device(device));
  auto sizes = state.sizes().vec();
  auto strides = state.strides().vec();
  strides[3] = c.Sq;
  strides[4] = c.layout_int + 1;
  return state.as_strided(sizes, strides);
}

static void validate_ring_state(const OnlineCtx& c, const at::Tensor& state, const at::Device& device) {
  if (!state.defined()) throw std::invalid_argument("invalid online state: tensor is undefined");
  const int nqb = (c.Sq + 383) / 384;
  const int64_t total_bytes = (int64_t)c.ring_bytes_fn(c.B, c.Hq, c.Sq);
  const int64_t block_bytes = total_bytes / ((int64_t)c.B * c.Hq * nqb);
  const bool valid_shape = state.dim() == 6 && state.size(0) == c.B && state.size(1) == c.Hq &&
      state.size(2) == nqb && state.size(3) == 1 && state.size(4) == 1 && state.size(5) == block_bytes;
  if (state.scalar_type() != at::kByte || !state.device().is_cuda() || state.device() != device ||
      !state.is_contiguous() || state.storage_offset() != 0 || state.numel() != total_bytes || !valid_shape ||
      state.stride(3) != c.Sq || state.stride(4) != c.layout_int + 1) {
    std::ostringstream oss;
    oss << "invalid online state for q: expected byte state metadata for batch=" << c.B
        << ", query_heads=" << c.Hq << ", query_length=" << c.Sq << ", layout=" << (c.bshd ? "bshd" : "bhsd")
        << ", device=" << device << "; got shape=" << state.sizes() << ", dtype=" << state.scalar_type()
        << ", device=" << state.device();
    throw std::invalid_argument(oss.str());
  }
}
// One chunk: V-transpose + fold into the running ring_state (Out written only when is_last).
static void online_run_chunk(
    const OnlineCtx& c, int Hkv, const at::Tensor& q, const at::Tensor& k_c, const at::Tensor& v_c, at::Tensor& out,
    at::Tensor& ring_state, bool is_first, bool is_last, void* stream, const SparseArgs* sparse = nullptr) {
  int len = v_c.size(c.sax);
  int len_pad = ((len + 63) / 64) * 64;
  at::Tensor v_t = at::empty({c.B, Hkv, len_pad, c.D}, v_c.options());
  int rc = c.vt_fn(v_c.data_ptr(), v_t.data_ptr(), c.B * Hkv * len, len, Hkv, c.layout_int, stream);
  if (rc != 0) throw std::runtime_error("online v_transpose rc=" + std::to_string(rc));
  rc = c.online_fn(
      q.data_ptr(), k_c.data_ptr(), v_t.data_ptr(), out.data_ptr(), c.B, c.Hq, Hkv, c.Sq, len, c.D, c.layout_int,
      ring_state.data_ptr(), (int)is_first, (int)is_last, stream, sparse);
  if (rc != 0) throw std::runtime_error("online chunk rc=" + std::to_string(rc));
}
}  // namespace

// Online (RingAttention) single chunk: fold k_c/v_c into the running state, return (out, state). Mirrors the reference
// online_flash_attn_func's per-chunk contract: is_first allocates state, non-first threads it back, is_last finalizes.
std::tuple<at::Tensor, at::Tensor> forward_online_chunk(
    at::Tensor q,
    at::Tensor k_c,
    at::Tensor v_c,
    bool is_first,
    bool is_last,
    std::optional<at::Tensor> state_,
    std::optional<at::Tensor> out_,
    std::string round_mode,
    std::string layout,
    std::vector<at::Tensor> sparse_tensors) {
  check_tensor(q, "q");
  check_tensor(k_c, "k", /*check_contiguous=*/false);  // KV chunk views are made contiguous internally
  check_tensor(v_c, "v", /*check_contiguous=*/false);
  if (q.device() != k_c.device() || q.device() != v_c.device())
    throw std::invalid_argument("q, k, v must be on the same device");
  OnlineCtx c = online_ctx(q, round_mode, layout);
  const int Hkv = validate_online_kv(c, k_c, v_c);
  const int Skv = k_c.size(c.sax);
  if (Skv <= 0) throw std::invalid_argument("KV chunk seq_len must be positive, got " + std::to_string(Skv));
  const c10::cuda::CUDAGuard device_guard(q.device());
  void* stream = (void*)at::cuda::getCurrentCUDAStream(q.device().index()).stream();

  at::Tensor ring_state;
  if (is_first) {
    if (state_.has_value()) throw std::invalid_argument("state must be None on the first online chunk (allocated here)");
    ring_state = allocate_ring_state(c, q.device());
  } else {
    if (!state_.has_value()) throw std::invalid_argument("state must be provided on non-first online chunks");
    ring_state = state_.value();
    validate_ring_state(c, ring_state, q.device());
  }
  SparseArgs sparse;
  if (!sparse_tensors.empty()) {
    if (sparse_tensors.size() != 8) throw std::invalid_argument("sparse requires counts, indices, types, q_aux, kv_aux, mask_offsets, mask_bits, q_order");
    for (const auto& t : sparse_tensors)
      if (t.scalar_type() != at::kInt || !t.is_contiguous() || t.device() != q.device())
        throw std::invalid_argument("sparse tensors must be contiguous int32 on q device");
    const int nq = (c.Sq + 383) / 384, nk = (Skv + 63) / 64;
    if (sparse_tensors[0].numel() != nq || sparse_tensors[1].numel() != nq * nk ||
        sparse_tensors[2].numel() != nq * nk || sparse_tensors[3].numel() != 4 * c.Sq ||
        sparse_tensors[4].numel() != 4 * Skv || sparse_tensors[5].numel() != nq * nk ||
        sparse_tensors[6].dim() != 3 || sparse_tensors[6].size(1) != 384 || sparse_tensors[6].size(2) != 2 ||
        sparse_tensors[7].numel() != nq)
      throw std::invalid_argument("sparse tensor shape mismatch");
    sparse = {sparse_tensors[0].data_ptr<int>(), sparse_tensors[1].data_ptr<int>(),
              sparse_tensors[2].data_ptr<int>(), sparse_tensors[3].data_ptr<int>(), sparse_tensors[4].data_ptr<int>(),
              sparse_tensors[5].data_ptr<int>(), sparse_tensors[6].data_ptr<int>(), sparse_tensors[7].data_ptr<int>()};
  }
  at::Tensor out = validate_or_allocate_out(q, out_);
  online_run_chunk(c, Hkv, q, k_c.contiguous(), v_c.contiguous(), out, ring_state, is_first, is_last, stream, sparse_tensors.empty() ? nullptr : &sparse);
  return {out, ring_state};
}

// Online convenience: split K/V into num_chunks contiguous chunks and thread the running state across per-chunk calls
// (is_first / is_last handled internally). Numerically identical to full attention. O(1) state in num_chunks.
at::Tensor forward_online(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    int num_chunks,
    std::optional<at::Tensor> out_,
    std::string round_mode,
    std::string layout) {
  check_tensor(q, "q"); check_tensor(k, "k"); check_tensor(v, "v");
  if (q.device() != k.device() || q.device() != v.device())
    throw std::invalid_argument("q, k, v must be on the same device");
  OnlineCtx c = online_ctx(q, round_mode, layout);
  const int Hkv = validate_online_kv(c, k, v);
  int Skv = k.size(c.sax);
  if (Skv <= 0) throw std::invalid_argument("Skv (KV seq_len) must be positive, got " + std::to_string(Skv));
  if (num_chunks < 1) throw std::invalid_argument("num_chunks must be >= 1");
  if (num_chunks > Skv) num_chunks = Skv;  // never produce an empty chunk

  at::Tensor out = validate_or_allocate_out(q, out_);
  const c10::cuda::CUDAGuard device_guard(q.device());
  void* stream = (void*)at::cuda::getCurrentCUDAStream(q.device().index()).stream();
  at::Tensor ring_state = allocate_ring_state(c, q.device());

  // ceil(Skv / num_chunks); the last non-empty chunk may be shorter. With a ceil chunk size, the KV
  // can be exhausted before i reaches num_chunks - 1 (e.g. Skv=6, num_chunks=4 → chunk=2 fills after
  // i=2), so is_last must key off "this chunk reaches the end", NOT i == num_chunks - 1. Otherwise the
  // real last chunk runs with is_last=false and the kernel never finalizes/normalizes/writes out.
  const int chunk = (Skv + num_chunks - 1) / num_chunks;
  for (int i = 0; i < num_chunks; ++i) {
    const int start = i * chunk;
    const int len = std::min(chunk, Skv - start);
    if (len <= 0) break;
    const bool is_last = (start + len >= Skv);
    at::Tensor k_c = k.narrow(c.sax, start, len).contiguous();
    at::Tensor v_c = v.narrow(c.sax, start, len).contiguous();
    online_run_chunk(c, Hkv, q, k_c, v_c, out, ring_state, /*is_first=*/i == 0, is_last, stream);
    if (is_last) break;
  }
  return out;
}

// LiteAttention forward: process only the K-blocks named in read_list, emit the next-step skip list
// into write_list. Mirrors forward() plus the int16 skip-list args (read/write/must_do).
at::Tensor forward_lite(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor read_list,
    at::Tensor write_list,
    float threshold,
    int phase,
    std::optional<at::Tensor> must_do_list_,
    std::optional<at::Tensor> out_,
    std::string round_mode,
    std::string layout) {
  check_tensor(q, "q");
  check_tensor(k, "k");
  check_tensor(v, "v");

  if (round_mode != "rtna" && round_mode != "rtne" && round_mode != "rtz") {
    throw std::invalid_argument("round_mode must be 'rtna', 'rtne', or 'rtz', got '" + round_mode + "'");
  }
  bool bshd = (layout == "bshd");
  if (!bshd && layout != "bhsd") {
    throw std::invalid_argument("layout must be 'bhsd' or 'bshd', got '" + layout + "'");
  }
  int layout_int = bshd ? 1 : 0;
  int sax = bshd ? 1 : 2;
  int hax = bshd ? 2 : 1;

  int B = q.size(0);
  int Hq = q.size(hax);
  int Hkv = k.size(hax);
  int Sq = q.size(sax);
  int D = q.size(3);
  int Skv = k.size(sax);
  if (k.size(0) != B || k.size(3) != D || v.sizes() != k.sizes()) {
    throw std::invalid_argument("LiteAttention k and v shapes must match and share q batch and head_dim");
  }
  if (Hq <= 0 || Hkv <= 0 || Hq < Hkv || Hq % Hkv != 0) {
    throw std::invalid_argument("GQA head mismatch in LiteAttention");
  }
  if (D != 128) {
    throw std::invalid_argument("head_dim must be 128, got " + std::to_string(D));
  }
  if (q.device() != k.device() || q.device() != v.device()) {
    throw std::invalid_argument("q, k, v must be on the same device");
  }

  // Skip lists must be contiguous int16 on q's device (must_do_list is optional).
  auto check_list = [&](const at::Tensor& t, const std::string& name) {
    if (t.scalar_type() != at::kShort) {
      throw std::invalid_argument(name + " must be int16, got " + std::string(at::toString(t.scalar_type())));
    }
    if (!t.is_contiguous()) {
      throw std::invalid_argument(name + " must be contiguous");
    }
    if (t.device() != q.device()) {
      throw std::invalid_argument(name + " must be on the same device as q");
    }
  };
  check_list(read_list, "read_list");
  check_list(write_list, "write_list");
  const void* must_do_ptr = nullptr;
  if (must_do_list_.has_value()) {
    check_list(must_do_list_.value(), "must_do_list");
    must_do_ptr = must_do_list_.value().data_ptr();
  }

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    check_tensor(out, "out");
    if (out.sizes() != q.sizes()) {
      throw std::invalid_argument("out shape must match q.shape");
    }
    if (out.device() != q.device()) {
      throw std::invalid_argument("out must be on the same device as q");
    }
  } else {
    out = at::empty_like(q);
  }

  const c10::cuda::CUDAGuard device_guard(q.device());
  void* stream = (void*)at::cuda::getCurrentCUDAStream(q.device().index()).stream();

  int Skv_pad = ((Skv + 63) / 64) * 64;
  at::Tensor v_t = at::empty({B, Hkv, Skv_pad, D}, v.options());

  int (*vt_fn)(const void*, void*, int, int, int, int, void*);
  int (*lite_fn)(
      const void*,
      const void*,
      const void*,
      void*,
      int,
      int,
      int,
      int,
      int,
      int,
      const void*,
      void*,
      const void*,
      float,
      int,
      int,
      void*,  // workspace (LPT order buffer)
      void*);
  size_t (*ltwsbytes_fn)(int, int, int);
  if (round_mode == "rtna") {
    vt_fn = launch_v_transpose_rtna;
    lite_fn = launch_attention_forward_lite_rtna;
    ltwsbytes_fn = lite_workspace_bytes_rtna;
  } else if (round_mode == "rtne") {
    vt_fn = launch_v_transpose_rtne;
    lite_fn = launch_attention_forward_lite_rtne;
    ltwsbytes_fn = lite_workspace_bytes_rtne;
  } else {  // rtz
    vt_fn = launch_v_transpose_rtz;
    lite_fn = launch_attention_forward_lite_rtz;
    ltwsbytes_fn = lite_workspace_bytes_rtz;
  }

  // LPT work-balance order buffer (default). OF4AMD_LITE_PLAIN=1 → null → plain identity-order grid.
  void* ws_ptr = nullptr;
  at::Tensor workspace;
  if (std::getenv("OF4AMD_LITE_PLAIN") == nullptr) {
    workspace = at::empty({(int64_t)ltwsbytes_fn(B, Hq, Sq)}, at::TensorOptions().dtype(at::kByte).device(q.device()));
    ws_ptr = workspace.data_ptr();
  }

  int rc = vt_fn(v.data_ptr(), v_t.data_ptr(), B * Hkv * Skv, Skv, Hkv, layout_int, stream);
  if (rc != 0) {
    throw std::runtime_error("launch_v_transpose returned error code " + std::to_string(rc));
  }
  rc = lite_fn(
      q.data_ptr(),
      k.data_ptr(),
      v_t.data_ptr(),
      out.data_ptr(),
      B,
      Hq,
      Hkv,
      Sq,
      Skv,
      D,
      read_list.data_ptr(),
      write_list.data_ptr(),
      must_do_ptr,
      threshold,
      phase,
      layout_int,
      ws_ptr,
      stream);
  if (rc != 0) {
    throw std::runtime_error("launch_attention_forward_lite returned error code " + std::to_string(rc));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "online_flash_4_amd: ROCm CDNA3 (MI300X) fused forward attention (+ online/ring path)";

  m.def(
      "forward",
      &forward,
      "Fused forward attention: O = softmax(QK^T / sqrt(D)) V",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("out") = py::none(),
      py::arg("round_mode") = "rtne",
      py::arg("layout") = "bshd");

  m.def(
      "forward_online",
      &forward_online,
      "Online (RingAttention) attention: thread running state across num_chunks contiguous KV chunks",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("num_chunks"),
      py::arg("out") = py::none(),
      py::arg("round_mode") = "rtne",
      py::arg("layout") = "bshd");

  m.def(
      "forward_online_chunk",
      &forward_online_chunk,
      "Online (RingAttention) single chunk -> (out, state). is_first allocates state, is_last finalizes.",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("is_first"),
      py::arg("is_last"),
      py::arg("state") = py::none(),
      py::arg("out") = py::none(),
      py::arg("round_mode") = "rtne",
      py::arg("layout") = "bshd",
      py::arg("sparse_tensors") = std::vector<at::Tensor>{});

  m.def(
      "forward_lite",
      &forward_lite,
      "LiteAttention forward with K-block skipping (not yet implemented)",
      py::arg("q"),
      py::arg("k"),
      py::arg("v"),
      py::arg("read_list"),
      py::arg("write_list"),
      py::arg("threshold"),
      py::arg("phase"),
      py::arg("must_do_list") = py::none(),
      py::arg("out") = py::none(),
      py::arg("round_mode") = "rtne",
      py::arg("layout") = "bshd");
}
