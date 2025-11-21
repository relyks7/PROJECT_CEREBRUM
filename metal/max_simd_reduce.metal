#include <metal_stdlib>
using namespace metal;
#define T 128
#define LANES 32
#define WARPS T/(LANES)
kernel void sum_simd_reduce(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_threadgroup]],
    uint j [[thread_position_in_grid]],
    uint k [[threadgroup_position_in_grid]],
    uint si [[thread_index_in_simdgroup]],
    uint sj [[simdgroup_index_in_threadgroup]]
) {
    float val=(j<n)?A[j]:-INFINITY;
    float local_sum=simd_max(val);
    threadgroup float pm[WARPS];
    if (si==0){
        pm[sj]=local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sj==0){
        float xm = (si < WARPS) ? pm[si] : -INFINITY;
        float final_max=simd_max(xm);
        if (si==0){
            B[k]=final_max;
        }
    }
}