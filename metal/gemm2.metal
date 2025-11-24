#include <metal_stdlib>
using namespace metal;
#define T 32
#define WIDTH 8
//Assume here that A is mxn and B is pxn
kernel void gemm2(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& p [[buffer(5)]],
    constant uint& b [[buffer(6)]],
    uint2 i [[thread_position_in_threadgroup]],
    uint3 j [[threadgroup_position_in_grid]],
    uint si [[simdgroup_index_in_threadgroup]]
)
{
    threadgroup float tA[T][T+1];
    threadgroup float tB_t[T][T+1];
    simdgroup_float8x8 acc; 
    simdgroup_float8x8 matA;
    simdgroup_float8x8 matB_t;
    acc = make_filled_simdgroup_matrix(0.0f);
    int diff=4;
    ushort2 offset = ushort2((si % diff) * WIDTH, (si/ diff) * WIDTH);
    uint row=j.y*T+i.y;
    uint col=j.x*T+i.x;
    uint layer=j.z;
    if (layer>=b) return;
    unsigned long offsetA = (unsigned long)layer * m * n;
    unsigned long offsetB = (unsigned long)layer * n * p;
    unsigned long offsetC = (unsigned long)layer * m * p;
    for (int curtile=0;curtile<(n+T-1)/T;curtile++){
        if (row < m && (curtile*T + i.x) < n)
            tA[i.y][i.x] = A[offsetA+row*n + (curtile*T + i.x)];
        else
            tA[i.y][i.x] = 0.0f;
        unsigned long Br=j.x*T+i.y;
        if (Br < p && (curtile*T + i.x) < n)
            tB_t[i.x][i.y] = B[offsetB+Br*n + (curtile*T + i.x)];
        else
            tB_t[i.x][i.y] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (si<16){
            #pragma unroll
            for (int k=0;k<T;k+=WIDTH){
                simdgroup_load(matA, (threadgroup float*)&tA[0][0], T+1, ulong2(k, offset.y));
                simdgroup_load(matB_t, (threadgroup float*)&tB_t[0][0], T+1, ulong2(offset.x, k));
                simdgroup_multiply_accumulate(acc, matA, matB_t, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint row0=j.y*T+offset.y;
    uint col0=j.x*T+offset.x;
    if (si<16){
        if (row0<m && col0<p){
            simdgroup_store(acc, C+offsetC, p, ulong2(col0, row0));
        }
    }
}