#include <metal_stdlib>
using namespace metal;
kernel void outer_product(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& m[[buffer(4)]],
    constant uint& b[[buffer(5)]],
    uint3 i[[thread_position_in_grid]]
){
    if (i.x<m && i.y<n && i.z<b){
        C[i.z*n*m+i.y*m+i.x]=A[i.z*n+i.y]*B[i.z*m+i.x];
    }
}