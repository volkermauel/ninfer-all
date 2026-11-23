// Inkling 2's quantized KV planes: kv_store and ring_load (kernels.h, kv_planes.h). One warp
// encodes one row of one head, one block decodes one ring row; the codings are the engine's D128
// variants of its KV codecs (kv_codec.cuh).

#include "models/inkling2/kv_planes.h"

#include "models/inkling2/kv_codec.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::kernels {
namespace {

constexpr int kWarps = 4; // warps per block, one (row, head) unit each

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2 ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

template <KvCoding C>
__launch_bounds__(kWarps * 32) __global__ void kv_store_kernel(KvStoreArgs a) {
    constexpr int kCode = kv_code_bytes(C), kScale = kv_scale_bytes(C);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int unit = blockIdx.x * kWarps + warp;
    if (unit >= a.rows_count * a.heads) { return; }
    const int head = unit % a.heads;
    const int r    = unit / a.heads;
    const int slot = physical(a.rows, r);
    if (a.end > a.begin && (slot < a.begin || slot >= a.end)) { return; }
    const int src_slot = a.src_row_slots ? r : (a.src_ring != 0 ? slot % a.src_ring : slot);
    const int dst_slot = a.dst_ring != 0 ? slot % a.dst_ring : slot;
    const int sequence = row_sequence(a.rows, r);
    const __half* src  = a.src + head * a.src_head_stride + std::int64_t(src_slot) * kKvRowDims;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { x[i] = __half2float(src[lane + 32 * i]); }
    // Paged planes address through the block table (dst_slot == slot there); ring planes are
    // never paged and take the dense fallback with the ring-modulo slot.
    std::uint8_t* codes  = a.dst.codes + kv_plane_offset(a.dst, sequence, head, dst_slot,
                                                         a.dst.code_seq, a.dst.code_head,
                                                         a.dst.code_row, a.dst.code_page);
    std::uint8_t* scales = a.dst.scales + kv_plane_offset(a.dst, sequence, head, dst_slot,
                                                          a.dst.scale_seq, a.dst.scale_head,
                                                          a.dst.scale_row, a.dst.scale_page);
    kv::encode_row<C>(x, codes, scales, lane);
}

// One warp per (position, head): lanes 0-15 decode the K row's sixteen chunks, 16-31 the V row's.
__global__ __launch_bounds__(32) void ring_load_kernel(RingLoadArgs a) {
    const int p    = a.begin + blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x;
    const bool key = lane < 16;
    const KvPlaneView& plane = key ? a.src_k : a.src_v;
    const int src = p % a.src_ring, dst = p % a.dst_ring, chunk = lane & 15;
    const std::int64_t codes_row = std::int64_t(a.sequence) * plane.code_seq +
                                   std::int64_t(h) * plane.code_head +
                                   std::int64_t(src) * plane.code_row;
    const std::int64_t scales_row = std::int64_t(a.sequence) * plane.scale_seq +
                                    std::int64_t(h) * plane.scale_head +
                                    std::int64_t(src) * plane.scale_row;
    const uint4 word = chunk_word(plane, plane.codes + codes_row, plane.scales + scales_row, chunk);
    __half* out = key ? a.dst_k : a.dst_v;
    reinterpret_cast<uint4*>(out + std::int64_t(h) * a.dst_head_stride +
                             std::int64_t(dst) * kKvRowDims)[chunk] = word;
}

} // namespace

void kv_store(const KvStoreArgs& args, cudaStream_t stream) {
    const int units = args.rows_count * args.heads;
    if (units <= 0) { return; }
    const dim3 grid((units + kWarps - 1) / kWarps), block(kWarps * 32);
    switch (args.dst.coding) {
    case KvCoding::Int8G64: kv_store_kernel<KvCoding::Int8G64><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::Int4G32: kv_store_kernel<KvCoding::Int4G32><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::Lloyd4: kv_store_kernel<KvCoding::Lloyd4><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::F16: throw std::invalid_argument("inkling2 kv_store: F16 needs no plane");
    }
    check_launch("kv_store");
}

void ring_load(const RingLoadArgs& args, cudaStream_t stream) {
    if (args.end <= args.begin) { return; }
    const dim3 grid(args.end - args.begin, args.kv_heads, 1);
    ring_load_kernel<<<grid, 32, 0, stream>>>(args);
    check_launch("ring_load");
}

} // namespace ninfer::models::inkling2::kernels
