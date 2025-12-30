#include <metal_stdlib>
using namespace metal;
kernel void really_specific_kernel(
    device float* G[[buffer(0)]],
    device const float* M[[buffer(1)]],
    device const float* ST[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant float& lambda[[buffer(4)]],
    uint i[[thread_position_in_grid]]
){
    if (i<n){
        G[i]=max(0.0f, lambda*(G[i]-M[0])/(ST[0] + 1e-6f));
    }
}