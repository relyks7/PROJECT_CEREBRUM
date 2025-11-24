#include <metal_stdlib>
using namespace metal;
#define T 32
#define WIDTH 8
//Assume A is nxm and B is nxp
kernel void gemm3(
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
    threadgroup float tA_t[T][T+1];
    threadgroup float tB[T][T+1];
    simdgroup_float8x8 acc; 
    simdgroup_float8x8 matA_t;
    simdgroup_float8x8 matB;
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
        unsigned long Ar = j.y*T+i.x;
        if (Ar < m && (curtile*T + i.y) < n)
            tA_t[i.x][i.y] = A[offsetA+(curtile*T + i.y)*m + Ar];
        else
            tA_t[i.x][i.y] = 0.0f;

        if ((curtile*T + i.y) < n && col < p)
            tB[i.y][i.x] = B[offsetB+(curtile*T + i.y)*p + col];
        else
            tB[i.y][i.x] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (si<16){
            #pragma unroll
            for (int k=0;k<T;k+=WIDTH){
                simdgroup_load(matA_t, (threadgroup float*)&tA_t[0][0], T+1, ulong2(k, offset.y));
                simdgroup_load(matB, (threadgroup float*)&tB[0][0], T+1, ulong2(offset.x, k));
                simdgroup_multiply_accumulate(acc, matA_t, matB, acc);
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