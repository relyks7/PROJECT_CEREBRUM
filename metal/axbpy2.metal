#include <metal_stdlib>
using namespace metal;
kernel void axbpy2(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant float& Y[[buffer(3)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        uint idx=i.y*n+i.x;
        C[idx]=A[idx]*(1-Y-l)+B[idx]*Y;
    }
}