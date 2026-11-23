// Inkling 2's paged full-attention planes (--kv-paging): the shared pool of page groups, the
// lanes' execution-table rows, and the kernels reading through them — against the dense planes,
// bit for bit:
//  - pool: leases follow uneven lane usage and return on reclaim; reservation admission refuses
//    over-subscription cleanly; execution-table rows publish and survive a lane's release;
//  - kv_store: a paged store lands byte for byte where the (permuted) table row names, matching
//    the dense store;
//  - attention: a prompt chunk, decode rows of two lanes and verification rows over paged planes
//    produce the dense run's outputs bit for bit (rk8v4 and rk4v4).

#include "core/device.h"
#include "core/device_buffer.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include "cuda_availability.h"
#include "models/inkling2/kernels.h"
#include "models/inkling2/kv_planes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
namespace k  = ninfer::models::inkling2::kernels;
namespace mi = ninfer::models::inkling2;
using bf16   = __nv_bfloat16;

constexpr int kHeads = 24, kKv = 8, kD = 128, kRel = 16;
constexpr int kExtent = 1024;
// Positions of one page group under test (main sweeps them); the core's default stays 64.
int g_page = kPagedKVPageSize;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int failures = 0;

void report(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

template <typename T>
DeviceBuffer upload(const std::vector<T>& values) {
    DeviceBuffer buffer(std::max<std::size_t>(values.size() * sizeof(T), 16));
    if (!values.empty()) { buffer.copy_from_host(values.data(), values.size() * sizeof(T)); }
    return buffer;
}

template <typename T>
std::vector<T> download(const DeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), count * sizeof(T));
    return values;
}

template <typename T>
T* as(DeviceBuffer& buffer) {
    return static_cast<T*>(buffer.p);
}

std::vector<float> normal(std::size_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) { x = d(rng); }
    return v;
}

// One visit's four planes (K codes, K scales, V codes, V scales), page-major, 8 KV heads.
KVPageGeometry visit_geometry(mi::KvCoding key, mi::KvCoding value) {
    KVPageGeometry geometry;
    geometry.page_tokens        = g_page;
    geometry.device_plane_order = PagedKVPlaneOrder::PageMajor;
    const auto plane = [&](DType dtype, std::int32_t leading) {
        return KVPlaneGeometry{dtype, leading, kKv, 256, 1, 0};
    };
    geometry.planes.push_back(
        plane(key == mi::KvCoding::Int8G64 ? DType::I8 : DType::U8, mi::kv_code_bytes(key)));
    geometry.planes.push_back(plane(DType::FP16, mi::kv_scale_bytes(key) / 2));
    geometry.planes.push_back(plane(DType::U8, mi::kv_code_bytes(value)));
    geometry.planes.push_back(plane(DType::FP16, mi::kv_scale_bytes(value) / 2));
    return geometry;
}

struct VisitSizes {
    int kc = 0, ks = 0, vc = 0, vs = 0;
};

VisitSizes visit_sizes(mi::KvCoding key, mi::KvCoding value) {
    return {mi::kv_code_bytes(key), mi::kv_scale_bytes(key), mi::kv_code_bytes(value),
            mi::kv_scale_bytes(value)};
}

struct Paged {
    DeviceKVPagePoolLayout pool_layout;
    KVExecutionTableLayout tables_layout;
    std::unique_ptr<DeviceArena> backing;
    std::unique_ptr<DeviceKVPagePool> pool;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::uint32_t lane_capacity = 0;
};

Paged make_paged(std::uint32_t groups, int lanes, std::uint32_t lane_capacity, mi::KvCoding key,
                 mi::KvCoding value) {
    Paged out;
    LayoutBuilder builder;
    out.pool_layout = plan_device_kv_page_pool(
        builder, {.page_group_count = groups, .geometry = visit_geometry(key, value)});
    out.tables_layout = plan_kv_execution_tables(
        builder, {.logical_page_capacity = lane_capacity, .table_rows = lanes});
    out.backing = std::make_unique<DeviceArena>(builder.finish(256));
    out.pool    = std::make_unique<DeviceKVPagePool>(
        DeviceSpan{out.backing->base(), out.backing->capacity()}, out.pool_layout);
    out.tables = std::make_unique<KVExecutionTablePool>(
        DeviceSpan{out.backing->base(), out.backing->capacity()}, out.tables_layout, *out.pool);
    out.lane_capacity = lane_capacity;
    return out;
}

// The call's rows (test_kv_planes.cpp's RowsCase), with a base sequence for single-lane fills.
struct RowsCase {
    std::vector<int> positions, sequences;
    int sequence = 0;
    bool chunk   = false;
    int first    = 0;
};

struct RowsDevice {
    DeviceBuffer positions, sequences;
    k::Rows rows;
    explicit RowsDevice(const RowsCase& c)
        : positions(upload(std::vector<std::int32_t>(c.positions.begin(), c.positions.end()))),
          sequences(upload(std::vector<std::int32_t>(c.sequences.begin(), c.sequences.end()))) {
        rows.positions = as<std::int32_t>(positions);
        rows.sequences = c.sequences.empty() ? nullptr : as<std::int32_t>(sequences);
        rows.sequence  = c.sequence;
        rows.tokens    = int(c.positions.size());
        rows.chunk     = c.chunk;
        rows.first     = c.first;
    }
};

mi::KvPlaneView dense_view(std::uint8_t* codes, std::uint8_t* scales, int slots, int cb, int sb,
                           mi::KvCoding coding) {
    mi::KvPlaneView view;
    view.codes      = codes;
    view.scales     = scales;
    view.code_row   = cb;
    view.scale_row  = sb;
    view.code_head  = std::int64_t(slots) * cb;
    view.scale_head = std::int64_t(slots) * sb;
    view.code_seq   = view.code_head * kKv;
    view.scale_seq  = view.scale_head * kKv;
    view.coding     = coding;
    return view;
}

mi::KvPlaneView paged_view(const Paged& paged, int plane_base, int cb, int sb, mi::KvCoding coding) {
    mi::KvPlaneView view;
    view.codes = static_cast<std::uint8_t*>(paged.pool->plane(std::size_t(plane_base)).data);
    view.scales = static_cast<std::uint8_t*>(paged.pool->plane(std::size_t(plane_base) + 1).data);
    view.code_row   = cb;
    view.scale_row  = sb;
    view.coding     = coding;
    view.block_table  = static_cast<const std::int32_t*>(paged.tables->matrix().data);
    view.table_stride = int(paged.lane_capacity);
    view.heads        = kKv;
    view.page_tokens  = g_page;
    view.code_page    = std::int64_t(g_page) * kKv * cb;
    view.scale_page   = std::int64_t(g_page) * kKv * sb;
    return view;
}

// kv_store of staged rows (one slot per row, kv_post's row-slots layout) into the dense planes
// (paged null) or into the pool.
void store_rows(const k::Rows& rows, const __half* src_k, const __half* src_v, int t,
                const VisitSizes& s, mi::KvCoding key, mi::KvCoding value, std::uint8_t* dense_kc,
                std::uint8_t* dense_ks, std::uint8_t* dense_vc, std::uint8_t* dense_vs, int slots,
                const Paged* paged) {
    const mi::KvCoding codings[2]  = {key, value};
    const int code_bytes[2]        = {s.kc, s.vc};
    const int scale_bytes[2]       = {s.ks, s.vs};
    std::uint8_t* dense_code[2]    = {dense_kc, dense_vc};
    std::uint8_t* dense_scale[2]   = {dense_ks, dense_vs};
    for (int plane = 0; plane < 2; ++plane) {
        k::KvStoreArgs store;
        store.rows            = rows;
        store.src             = plane == 0 ? src_k : src_v;
        store.src_head_stride = std::int64_t(t) * kD;
        store.rows_count      = t;
        store.heads           = kKv;
        store.dst             = paged != nullptr
                                    ? paged_view(*paged, plane * 2, code_bytes[plane],
                                                 scale_bytes[plane], codings[plane])
                                    : dense_view(dense_code[plane], dense_scale[plane], slots,
                                                 code_bytes[plane], scale_bytes[plane],
                                                 codings[plane]);
        k::kv_store(store, nullptr);
    }
}

// The attention's inputs for `t` rows, and one attention run into `out` (bf16).
struct AttnInputs {
    DeviceBuffer dq, dr, dt, dg, ds, partial, out;
    std::int64_t gate_stride = 0, sink_stride = 0;
};

AttnInputs make_attn_inputs(int t, int splits, std::mt19937& rng) {
    AttnInputs in;
    in.gate_stride = kHeads * kD + 32;
    in.sink_stride = kHeads + 3;
    in.dq          = upload(normal(std::size_t(t) * kHeads * kD, 0.6f, rng));
    in.dr          = upload(normal(std::size_t(t) * kHeads * kRel, 1.0f, rng));
    in.dt          = upload(normal(std::size_t(kExtent) * kRel, 1.0f, rng));
    in.dg          = upload(normal(std::size_t(t) * std::size_t(in.gate_stride), 1.0f, rng));
    in.ds          = upload(normal(std::size_t(t) * std::size_t(in.sink_stride), 2.0f, rng));
    in.partial     = DeviceBuffer(std::size_t(k::attention_workspace_floats(t, kHeads, splits)) * 4);
    in.out         = DeviceBuffer(std::size_t(t) * kHeads * kD * 2);
    return in;
}

void run_attention(AttnInputs& in, const k::Rows& rows, int t, int splits,
                   const mi::KvPlaneView& k_plane, const mi::KvPlaneView& v_plane) {
    k::AttentionArgs a;
    a.rows        = rows;
    a.q           = as<float>(in.dq);
    a.r           = as<float>(in.dr);
    a.sinks       = as<float>(in.ds);
    a.sink_stride = in.sink_stride;
    a.table       = as<float>(in.dt);
    a.extent      = kExtent;
    a.k_plane     = k_plane;
    a.v_plane     = v_plane;
    a.window      = 0;
    a.query_heads = kHeads;
    a.kv_heads    = kKv;
    a.scale       = 1.0f / std::sqrt(float(kD));
    a.splits      = splits;
    a.partial     = as<float>(in.partial);
    a.gate        = as<float>(in.dg);
    a.gate_stride = in.gate_stride;
    a.out         = as<bf16>(in.out);
    k::attention(a, nullptr);
}

// A lane's published table row, read back from the device.
std::vector<std::int32_t> table_row(const Paged& paged, const KVExecutionRowLease& row) {
    const Tensor tensor = paged.tables->row(row.handle());
    std::vector<std::int32_t> entries(paged.lane_capacity);
    check(cudaMemcpy(entries.data(), tensor.data, entries.size() * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost),
          "table row read");
    return entries;
}

// ---- The pool: leases, uneven lanes, admission, tables ------------------------------------------

void exercise_pool() {
    Paged paged = make_paged(8, 3, 4, mi::KvCoding::Int8G64, mi::KvCoding::Int4G32);
    DeviceKVPagePool& pool = *paged.pool;

    // Admission: an over-subscribed grant is refused cleanly, the pool unchanged.
    auto r0 = pool.reserve(6);
    report(r0.has_value(), "admission: a 6-group grant fits");
    const auto refused = pool.reserve(4);
    report(!refused.has_value(), "admission: a 4-group grant beside it is refused");
    report(pool.reserved_pages() == 6 && pool.allocated_pages() == 0,
           "admission: the refusal leaves the pool's counters alone");

    // Uneven lanes: 6 + 2 groups leased, the pool follows usage, not lanes x max.
    std::vector<DeviceKVPageLease> l0, l1;
    l0.reserve(6);
    pool.materialize(*r0, 6, l0);
    report(pool.allocated_pages() == 6 && l0.size() == 6, "lanes: six groups leased");
    auto r1b = pool.reserve(2);
    report(r1b.has_value(), "admission: a 2-group grant fits beside the six");
    l1.reserve(2);
    pool.materialize(*r1b, 2, l1);
    report(pool.allocated_pages() == 8 && pool.available_pages() == 0,
           "lanes: eight leased groups follow the lanes' uneven usage");

    // Reclaim: a lane trimmed to one group returns five, a retiring lane the rest.
    pool.dematerialize(*r0, 1, l0);
    report(pool.allocated_pages() == 3 && l0.size() == 1,
           "reclaim: a lane trimmed to one group returns the others");
    pool.dematerialize(*r0, 0, l0);
    r0->release();
    pool.dematerialize(*r1b, 0, l1);
    r1b->release();
    const auto runs = pool.free_runs();
    report(pool.allocated_pages() == 0 && runs.size() == 1 && runs[0].begin == 0 &&
               runs[0].count == 8,
           "reclaim: everything released joins one free run");

    // Execution tables: a row publishes its groups, and a released row serves a new lane. A fresh
    // pool hands out 0, 1, 2..., so the published ids are known.
    auto r2 = pool.reserve(3);
    std::vector<DeviceKVPageLease> l2;
    l2.reserve(3);
    pool.materialize(*r2, 3, l2);
    std::vector<DeviceKVPageHandle> handles;
    for (const DeviceKVPageLease& lease : l2) { handles.push_back(lease.handle()); }
    {
        KVExecutionRowLease row = paged.tables->acquire(0);
        paged.tables->publish(row.handle(), 0, handles);
        check(cudaDeviceSynchronize(), "table publish");
        const auto entries = table_row(paged, row);
        bool match = entries.size() >= 3;
        for (std::size_t i = 0; match && i < 3; ++i) { match = entries[i] == int(i); }
        report(match, "tables: a row publishes its lane's page groups in order");
        row.release();
    }
    {
        KVExecutionRowLease row = paged.tables->acquire(0);
        std::vector<DeviceKVPageHandle> reversed(handles.rbegin(), handles.rend());
        paged.tables->publish(row.handle(), 0, reversed);
        check(cudaDeviceSynchronize(), "table republish");
        const auto entries = table_row(paged, row);
        bool match = entries.size() >= 3;
        for (std::size_t i = 0; match && i < 3; ++i) { match = entries[i] == int(2 - i); }
        report(match, "tables: a released row serves the next lane's mapping");
        row.release();
    }
}

// ---- The kernels: paged stores and attention against the dense planes ---------------------------

// Compares every stored row's codes and scales: the dense planes (lane-major [slot][head]) against
// the pool at the (permuted) table row's groups.
void compare_stored(const char* name, const Paged& paged,
                    const std::vector<std::vector<std::int32_t>>& published, const VisitSizes& s,
                    DeviceBuffer& dense_kc, DeviceBuffer& dense_ks, DeviceBuffer& dense_vc,
                    DeviceBuffer& dense_vs, int lanes, int fill_slots) {
    const struct {
        int cb, sb;
        bool key;
    } pair[2]         = {{s.kc, s.ks, true}, {s.vc, s.vs, false}};
    DeviceBuffer* dense_code[2]  = {&dense_kc, &dense_vc};
    DeviceBuffer* dense_scale[2] = {&dense_ks, &dense_vs};
    for (int plane = 0; plane < 2; ++plane) {
        const int cb = pair[plane].cb, sb = pair[plane].sb;
        const auto dense_codes = download<std::uint8_t>(
            *dense_code[plane], std::size_t(lanes) * fill_slots * kKv * std::size_t(cb));
        const auto dense_scales = download<std::uint8_t>(
            *dense_scale[plane], std::size_t(lanes) * fill_slots * kKv * std::size_t(sb));
        const Tensor& codes_plane  = paged.pool->plane(std::size_t(plane * 2));
        const Tensor& scales_plane = paged.pool->plane(std::size_t(plane * 2 + 1));
        std::vector<std::uint8_t> pool_codes(codes_plane.bytes());
        check(cudaMemcpy(pool_codes.data(), codes_plane.data, pool_codes.size(),
                         cudaMemcpyDeviceToHost),
              "pool codes read");
        std::vector<std::uint8_t> pool_scales(scales_plane.bytes());
        check(cudaMemcpy(pool_scales.data(), scales_plane.data, pool_scales.size(),
                         cudaMemcpyDeviceToHost),
              "pool scales read");
        const std::size_t page_code_bytes   = std::size_t(g_page) * kKv * std::size_t(cb);
        const std::size_t page_scale_bytes  = std::size_t(g_page) * kKv * std::size_t(sb);
        for (int lane = 0; lane < lanes; ++lane) {
            for (int slot = 0; slot < fill_slots; ++slot) {
                const std::int32_t group =
                    published[std::size_t(lane)][std::size_t(slot / g_page)];
                const int in_group = slot % g_page;
                for (int head = 0; head < kKv; ++head) {
                    // The dense planes' layout is [sequence][head][slot].
                    const std::size_t dense_at =
                        (std::size_t(lane) * kKv * fill_slots + std::size_t(head) * fill_slots +
                         std::size_t(slot)) * std::size_t(cb);
                    const std::size_t paged_at =
                        std::size_t(group) * page_code_bytes +
                        (std::size_t(in_group) * kKv + std::size_t(head)) * std::size_t(cb);
                    if (std::memcmp(dense_codes.data() + dense_at, pool_codes.data() + paged_at,
                                    std::size_t(cb)) != 0) {
                        report(false, std::string(name) + ": codes of lane " +
                                          std::to_string(lane) + " slot " +
                                          std::to_string(slot) + " head " +
                                          std::to_string(head) + " differ from the dense store");
                        return;
                    }
                    const std::size_t dense_scale_at =
                        (std::size_t(lane) * kKv * fill_slots + std::size_t(head) * fill_slots +
                         std::size_t(slot)) * std::size_t(sb);
                    const std::size_t paged_scale_at =
                        std::size_t(group) * page_scale_bytes +
                        (std::size_t(in_group) * kKv + std::size_t(head)) * std::size_t(sb);
                    if (std::memcmp(dense_scales.data() + dense_scale_at,
                                    pool_scales.data() + paged_scale_at,
                                    std::size_t(sb)) != 0) {
                        report(false, std::string(name) + ": scales of lane " +
                                          std::to_string(lane) + " slot " +
                                          std::to_string(slot) + " head " +
                                          std::to_string(head) + " differ from the dense store");
                        return;
                    }
                }
            }
        }
    }
    report(true, std::string(name) + ": paged stores match the dense planes byte for byte");
}

// The attention runs of one call shape, dense and paged from the same inputs; the outputs' bits
// must agree.
void compare_attention(const char* name, const char* kind, const Paged& paged, const VisitSizes& s,
                       mi::KvCoding key, mi::KvCoding value, DeviceBuffer& dense_kc,
                       DeviceBuffer& dense_ks, DeviceBuffer& dense_vc, DeviceBuffer& dense_vs,
                       int slots, const RowsCase& rc, int splits, const std::vector<__half>& ks,
                       const std::vector<__half>& vs, std::mt19937& rng) {
    const int t = int(rc.positions.size());
    RowsDevice rows_dev(rc);
    DeviceBuffer src_k = upload(ks), src_v = upload(vs);
    AttnInputs in = make_attn_inputs(t, splits, rng);
    std::vector<std::uint16_t> outs[2];
    for (int which = 0; which < 2; ++which) {
        const bool is_paged = which == 1;
        store_rows(rows_dev.rows, as<__half>(src_k), as<__half>(src_v), t, s, key, value,
                   as<std::uint8_t>(dense_kc), as<std::uint8_t>(dense_ks),
                   as<std::uint8_t>(dense_vc), as<std::uint8_t>(dense_vs), slots,
                   is_paged ? &paged : nullptr);
        check(cudaMemset(in.out.p, 0xAB, in.out.bytes), "out clear");
        run_attention(in, rows_dev.rows, t, splits,
                      is_paged ? paged_view(paged, 0, s.kc, s.ks, key)
                               : dense_view(as<std::uint8_t>(dense_kc), as<std::uint8_t>(dense_ks),
                                            slots, s.kc, s.ks, key),
                      is_paged ? paged_view(paged, 2, s.vc, s.vs, value)
                               : dense_view(as<std::uint8_t>(dense_vc), as<std::uint8_t>(dense_vs),
                                            slots, s.vc, s.vs, value));
        check(cudaDeviceSynchronize(), kind);
        outs[std::size_t(which)] = download<std::uint16_t>(in.out, std::size_t(t) * kHeads * kD);
    }
    report(outs[0] == outs[1],
           std::string(name) + ": " + kind + " rows match the dense run bit for bit");
}

// One visit's codings: the fills, the byte-for-byte store comparison, and the three attention
// shapes, dense against paged.
void case_run(const char* name, mi::KvCoding key, mi::KvCoding value, std::mt19937& rng) {
    const VisitSizes s = visit_sizes(key, value);
    const int lanes = 2, fill_slots = 3 * g_page;
    const std::uint32_t groups_per_lane = std::uint32_t(fill_slots / g_page);

    Paged paged = make_paged(2 * groups_per_lane, lanes, groups_per_lane, key, value);
    DeviceBuffer dense_kc(std::size_t(lanes) * fill_slots * kKv * std::size_t(s.kc)),
        dense_ks(std::size_t(lanes) * fill_slots * kKv * std::size_t(s.ks)),
        dense_vc(std::size_t(lanes) * fill_slots * kKv * std::size_t(s.vc)),
        dense_vs(std::size_t(lanes) * fill_slots * kKv * std::size_t(s.vs));

    // The lanes' leases; their table rows permute the groups (pages relocated, not just indexed).
    // A fresh pool hands out groups in order, so lane 0's row is [2, 1, 0] and lane 1's [4, 3, 5].
    std::vector<std::vector<DeviceKVPageLease>> leases(lanes);
    std::vector<std::vector<DeviceKVPageHandle>> handles(lanes);
    std::vector<std::optional<DeviceKVPageReservation>> grants(lanes);
    std::vector<KVExecutionRowLease> execution_rows(lanes);
    for (int lane = 0; lane < lanes; ++lane) {
        grants[std::size_t(lane)] = paged.pool->reserve(groups_per_lane);
        if (!grants[std::size_t(lane)]) {
            report(false, std::string(name) + ": lane grant");
            return;
        }
        leases[std::size_t(lane)].reserve(groups_per_lane);
        paged.pool->materialize(*grants[std::size_t(lane)], groups_per_lane,
                                leases[std::size_t(lane)]);
        for (const DeviceKVPageLease& lease : leases[std::size_t(lane)]) {
            handles[std::size_t(lane)].push_back(lease.handle());
        }
        std::swap(handles[std::size_t(lane)][0], handles[std::size_t(lane)][2]);
        execution_rows[std::size_t(lane)] = paged.tables->acquire(lane);
        paged.tables->publish(execution_rows[std::size_t(lane)].handle(), 0,
                              handles[std::size_t(lane)]);
    }
    check(cudaDeviceSynchronize(), "table publish");
    std::vector<std::vector<std::int32_t>> published(lanes);
    for (int lane = 0; lane < lanes; ++lane) {
        published[std::size_t(lane)] = table_row(paged, execution_rows[std::size_t(lane)]);
        const std::vector<std::int32_t> want = {
            std::int32_t(lane * groups_per_lane + 2), std::int32_t(lane * groups_per_lane + 1),
            std::int32_t(lane * groups_per_lane + 0)};
        if (published[std::size_t(lane)] != want) {
            report(false, std::string(name) + ": lane " + std::to_string(lane) +
                              "'s table row does not hold the permuted groups");
            return;
        }
    }
    report(true, std::string(name) + ": the lanes' table rows hold the permuted groups");

    // Both lanes' fills: one chunk each, stored dense and paged from the same staged halves.
    for (int lane = 0; lane < lanes; ++lane) {
        RowsCase rc;
        rc.chunk    = true;
        rc.first    = 0;
        rc.sequence = lane;
        for (int i = 0; i < fill_slots; ++i) { rc.positions.push_back(i); }
        RowsDevice rows_dev(rc);
        std::vector<__half> ks(std::size_t(kKv) * fill_slots * kD),
            vs(std::size_t(kKv) * fill_slots * kD);
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : ks) { v = __float2half(1.5f * n(rng)); }
        for (auto& v : vs) { v = __float2half(n(rng)); }
        for (int which = 0; which < 2; ++which) {
            store_rows(rows_dev.rows, ks.data(), vs.data(), fill_slots, s, key, value,
                       as<std::uint8_t>(dense_kc), as<std::uint8_t>(dense_ks),
                       as<std::uint8_t>(dense_vc), as<std::uint8_t>(dense_vs), fill_slots,
                       which == 1 ? &paged : nullptr);
        }
    }
    check(cudaDeviceSynchronize(), "chunk fills");
    compare_stored(name, paged, published, s, dense_kc, dense_ks, dense_vc, dense_vs, lanes,
                   fill_slots);

    // The chunk's attention: lane 0's 192 rows over their stored keys.
    {
        RowsCase rc;
        rc.chunk    = true;
        rc.first    = 0;
        rc.sequence = 0;
        for (int i = 0; i < fill_slots; ++i) { rc.positions.push_back(i); }
        std::vector<__half> ks(std::size_t(kKv) * fill_slots * kD),
            vs(std::size_t(kKv) * fill_slots * kD);
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : ks) { v = __float2half(1.5f * n(rng)); }
        for (auto& v : vs) { v = __float2half(n(rng)); }
        compare_attention(name, "the chunk's", paged, s, key, value, dense_kc, dense_ks, dense_vc,
                          dense_vs, fill_slots, rc, 1, ks, vs, rng);
    }
    // Decode rows of both lanes, then verification rows past them.
    {
        RowsCase rc;
        rc.positions = {70, 140};
        rc.sequences = {0, 1};
        std::vector<__half> ks(std::size_t(kKv) * 2 * kD), vs(std::size_t(kKv) * 2 * kD);
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : ks) { v = __float2half(1.5f * n(rng)); }
        for (auto& v : vs) { v = __float2half(n(rng)); }
        compare_attention(name, "decode", paged, s, key, value, dense_kc, dense_ks, dense_vc,
                          dense_vs, fill_slots, rc, 4, ks, vs, rng);
    }
    {
        RowsCase rc;
        rc.positions = {71, 72, 73, 141, 142, 143};
        rc.sequences = {0, 0, 0, 1, 1, 1};
        std::vector<__half> ks(std::size_t(kKv) * 6 * kD), vs(std::size_t(kKv) * 6 * kD);
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : ks) { v = __float2half(1.5f * n(rng)); }
        for (auto& v : vs) { v = __float2half(n(rng)); }
        compare_attention(name, "verification", paged, s, key, value, dense_kc, dense_ks, dense_vc,
                          dense_vs, fill_slots, rc, 4, ks, vs, rng);
    }

    for (KVExecutionRowLease& row : execution_rows) { row.release(); }
}

} // namespace

int main() {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (ninfer::test::cuda_unavailable(error) || (error == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP  no CUDA device\n";
        return 77;
    }
    check(cudaSetDevice(0), "device");
    try {
        exercise_pool();
        // The landed 64-token groups and the sweep's other power-of-two divisors of the window:
        // the tables and both kernels must hold bit for bit at every group size.
        for (const int page : {kPagedKVPageSize, 32, 128, 1024}) {
            g_page = page;
            std::cout << "-- page groups of " << page << " positions\n";
            std::mt19937 rng(20261121);
            case_run("rk8v4 (int8 G64 keys, packed int4 G32 values)", mi::KvCoding::Int8G64,
                     mi::KvCoding::Int4G32, rng);
            case_run("rk4v4 (Lloyd-Max keys, packed int4 G32 values)", mi::KvCoding::Lloyd4,
                     mi::KvCoding::Int4G32, rng);
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL  " << e.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "PASS  paging\n" : "FAIL  paging\n");
    return failures == 0 ? 0 : 1;
}
