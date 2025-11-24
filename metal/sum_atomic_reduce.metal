#include <metal_stdlib>
using namespace metal;
kernel void sum_atomic_reduce(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& b[[buffer(3)]],
    uint2 i[[thread_position_in_grid]]
) {
    if (i.x<n && i.y<b){
        atomic_fetch_add_explicit(&B[i.y], A[i.y*n+i.x], memory_order_relaxed);
    }
}