#include <metal_stdlib>
using namespace metal;
kernel void max_atomic_reduce(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]],
) {
    if (i<n){
        atomic_fetch_max_explicit(&B[0], A[i], memory_order_relaxed);
    }
}