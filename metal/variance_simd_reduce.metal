#include <metal_stdlib>
using namespace metal;
#define T 128
#define LANES 32
#define WARPS T/(LANES)
kernel void variance_simd_reduce(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant float& mu [[buffer(3)]],
    uint i [[thread_position_in_threadgroup]],
    uint j [[thread_position_in_grid]],
    uint k [[threadgroup_position_in_grid]],
    uint si [[thread_index_in_simdgroup]],
    uint sj [[simdgroup_index_in_threadgroup]]
) {
    float val=(j<n)?(A[j]-mu)*(A[j]-mu)/n:0.0f;
    float local_sum=simd_sum(val);
    threadgroup float ps[WARPS];
    if (si==0){
        ps[sj]=local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sj==0){
        float xs = (si < WARPS) ? ps[si] : 0.0f;
        float final_sum=simd_sum(xs);
        if (si==0){
            B[k]=final_sum;
        }
    }
}