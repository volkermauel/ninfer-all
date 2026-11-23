#pragma once

// Inkling 2's KV-cache storage (--kv-dtype). A cache row is one KV head of one position, its
// 128 values in one block: codes and scales in two planes of a visit (executor.cpp). The codings
// are the engine's own codecs (ops/kv_cache/*_codec.cuh) over groups of the row:
//
//   Int8G64   signed int8, one FP16 scale per 64 dimensions, FP16-RNE(absmax / 127)
//   Int4G32   two signed 4-bit codes per byte, one FP16 scale per 32, FP16-RNE(absmax / 7)
//   Lloyd4    4-bit Lloyd-Max indices with a G64 FP16 least-squares scale (rk4v4 keys)
//
// The visits' caches quantize alike (rk8v4: Int8G64 keys and Int4G32 values; rk4v4: Lloyd4 keys
// and Int4G32 values): the eight full-attention visits' elastic planes and the forty windowed
// visits' fixed sliding rings. The convolution histories and the DSpark drafter's caches stay F16
// whatever the flag says. Unlike Gemma 4's sliding blocks the
// rows here are stored unrotated: Inkling's K and V come out of their convolutions clipped and
// RMS-normed, and Gemma 4's full-attention blocks store the same codings unrotated (its rk8v4
// quality numbers cover exactly this). The codings otherwise need no rotation to reach their
// measured error.

#include "ninfer/kv_cache_storage.h"

#include <cstdint>

// The sizes below are also compile-time constants of the CUDA kernels.
#if defined(__CUDACC__)
#    define NINFER_INKLING2_KV_HOST_DEVICE __host__ __device__
#else
#    define NINFER_INKLING2_KV_HOST_DEVICE
#endif

namespace ninfer::models::inkling2 {

enum class KvCoding : std::uint8_t { F16, Int8G64, Int4G32, Lloyd4 };

// One row: one KV head's 128 values, so 128-wide blocks.
inline constexpr int kKvRowDims = 128;

struct KvPlane {
    KvCoding coding = KvCoding::F16;
};

struct KvFormat {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    KvPlane key, value;
    [[nodiscard]] bool quantized() const noexcept {
        return storage != KvCacheStorage::BFloat16 && storage != KvCacheStorage::Float16;
    }
};

[[nodiscard]] KvFormat kv_format(KvCacheStorage storage);

// Bytes of one row's codes and of its scales.
[[nodiscard]] NINFER_INKLING2_KV_HOST_DEVICE constexpr int kv_code_bytes(KvCoding c) noexcept {
    switch (c) {
    case KvCoding::F16: return 2 * kKvRowDims;
    case KvCoding::Int8G64: return kKvRowDims;
    case KvCoding::Int4G32:
    case KvCoding::Lloyd4: return kKvRowDims / 2;
    }
    return 0;
}

[[nodiscard]] NINFER_INKLING2_KV_HOST_DEVICE constexpr int kv_scale_bytes(KvCoding c) noexcept {
    switch (c) {
    case KvCoding::F16: return 0;
    case KvCoding::Int8G64:
    case KvCoding::Lloyd4: return 2 * 2; // two groups of 64
    case KvCoding::Int4G32: return 4 * 2; // four groups of 32
    }
    return 0;
}

// A quantized plane of a visit: one row's codes at codes + sequence * code_seq +
// head * code_head + slot * code_row, its scales likewise (executor.cpp's code and scale areas).
// A paged plane (--kv-paging, the shared pool of the eight full-attention visits) has no
// per-sequence strides: the row of slot lives in the page group named by the lane's row of the
// execution table (block_table), at (slot within the group) * heads + head rows into the group.
struct KvPlaneView {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    std::int64_t code_row = 0, scale_row = 0, code_head = 0, scale_head = 0, code_seq = 0,
                 scale_seq = 0;
    KvCoding coding = KvCoding::F16;
    // Null: dense addressing above. Else the device execution table ([lanes][table_stride]
    // groups, table_stride entries a lane row) and the plane's bytes per page group.
    const std::int32_t* block_table = nullptr;
    std::int64_t table_stride = 0, code_page = 0, scale_page = 0;
    int heads = 0;       // the plane's head extent (a page group holds page_tokens rows per head)
    int page_tokens = 0; // kPagedKVPageSize when paged
};

// Where a plane's row of (sequence, head, slot) begins: codes with the code strides and code
// page bytes, scales with the scale ones.
[[nodiscard]] NINFER_INKLING2_KV_HOST_DEVICE inline std::int64_t kv_plane_offset(
    const KvPlaneView& plane, int sequence, int head, int slot, std::int64_t seq_stride,
    std::int64_t head_stride, std::int64_t row_stride, std::int64_t page_bytes) {
    if (plane.block_table != nullptr) {
        const std::int32_t group = plane.block_table[std::int64_t(sequence) * plane.table_stride +
                                                     slot / plane.page_tokens];
        return std::int64_t(group) * page_bytes +
               std::int64_t((slot % plane.page_tokens) * plane.heads + head) * row_stride;
    }
    return std::int64_t(sequence) * seq_stride + std::int64_t(head) * head_stride +
           std::int64_t(slot) * row_stride;
}

} // namespace ninfer::models::inkling2
