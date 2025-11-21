#include <metal_stdlib>
using namespace metal;
#define T 16
#define WIDTH 8
kernel void gemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint2 j [[threadgroup_position_in_grid]],
    uint2 si [[simd_position_in_grid]]
)
{
    threadgroup float tA[T][T+1];
    threadgroup float tB[T][T+1];
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    uint row0=
    uint col0=
    float acc=0;
    for (int curtile=0;curtile<(n+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < n)
            tA[i.y][i.x] = A[row*n + curtile*T + i.x];
        else
            tA[i.y][i.x] = 0.0f;

        if ((curtile*T + i.y) < n && col < p)
            tB[i.y][i.x] = B[(curtile*T + i.y)*p + col];
        else
            tB[i.y][i.x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}