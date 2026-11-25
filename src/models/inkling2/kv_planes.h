#pragma once

// Inkling 2's quantized KV planes: storing attention_inputs' (upstream) or kv_post's (paged
// branch) F16 rows into them (kv_codec.cuh encodes one row per warp), and — a prompt chunk's
// window before the chunk — decoding a ring's rows back to F16. Reading the planes back
// otherwise happens in the attention's tile loader (kernels.h), which decodes eight-value chunks
// as it loads a tile — there is no F16 copy of a quantized cache.

#include "models/inkling2/kernels.h"
#include "models/inkling2/kv_format.h"

namespace ninfer::models::inkling2::kernels {

// Stores F16 rows into a quantized plane: row r's values, head h at src + h * src_head_stride +
// Stores F16 rows into a quantized plane: row r's values, head h at src + h * src_head_stride +
// r * 128 (KvArgs's `row_slots` layout, kv_post's staging scratch; without `src_row_slots` the
// rows come from a ring instead, slot physical(rows, r) % src_ring, 0: physical — a staging
// ring's kept rows), head h's plane row of position physical(rows, r) at slot physical(rows, r)
// (a full-attention visit: ring 0) of sequence row_sequence(rows, r). `dst_ring` stores them to
// a ring (a sliding visit's planes: slot physical(rows, r) % dst_ring), and [begin, end) keeps
// the store to the rows whose physical positions lie in it (both 0: every row).
struct KvStoreArgs {
    Rows rows;
    const __half* src = nullptr;
    std::int64_t src_head_stride = 0;
    bool src_row_slots = true;
    int src_ring = 0, dst_ring = 0, begin = 0, end = 0;
    int rows_count = 0, heads = 0;
    KvPlaneView dst;
};
void kv_store(const KvStoreArgs& args, cudaStream_t stream);

// Decodes the K and V rows of positions [begin, end) of one sequence from a quantized ring's
// planes into F16 rows of another ring: position p lives in slot p % src_ring of the planes and
// p % dst_ring of the destination ([kv heads][slots][128] halves, heads dst_head_stride elements
// apart) — a prompt chunk's window before the chunk, which its attention then reads as F16.
struct RingLoadArgs {
    KvPlaneView src_k, src_v;
    __half* dst_k = nullptr;
    __half* dst_v = nullptr;
    std::int64_t dst_head_stride = 0;
    int src_ring = 0, dst_ring = 0, kv_heads = 0, begin = 0, end = 0;
    int sequence = 0;
};
void ring_load(const RingLoadArgs& args, cudaStream_t stream);

} // namespace ninfer::models::inkling2::kernels
