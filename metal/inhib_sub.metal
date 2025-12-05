#include <metal_stdlib>
using namespace metal;
kernel void inhib_sub(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant float& alpha[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    constant uint& b[[buffer(5)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<n && i.y<b){
        C[i.y*n+i.x]=A[i.y*n+i.x]-alpha*B[i.x];
    }
}