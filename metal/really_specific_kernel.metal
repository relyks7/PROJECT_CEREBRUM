#include <metal_stdlib>
using namespace metal;
kernel void really_specific_kernel(
    device const float* A[[buffer(0)]],
    device const float* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant float& gamma[[buffer(4)]],
    uint i[[thread_position_in_grid]]
){
    if (i<n){
        C[i]=max(0.0f, A[i]-B[0]-gamma)/(1e-6+B[0]);
    }
}