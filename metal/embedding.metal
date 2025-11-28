#include <metal_stdlib>
using namespace metal;
kernel void embedding(
    device const float* A[[buffer(0)]],
    device const int* B[[buffer(1)]],
    device float* C[[buffer(2)]],
    constant uint& n[[buffer(3)]],
    constant uint& d[[buffer(4)]],
    constant uint& b[[buffer(5)]],
    uint i [[thread_position_in_grid]]
)
{
    if (i>=n) return;
    for (int j=0;j<d;j++){
        C[i*d+j]=A[B[i]*d+j];
    }
}