#include <metal_stdlib>
using namespace metal;
kernel void cortex_step(
    device const float* E_t[[buffer(0)]],
    device float* H_t1[[buffer(1)]],
    device const float* X_g[[buffer(2)]],
    device const float* X_m[[buffer(3)]],
    device const float* mu[[buffer(4)]],
    device const float* gamma[[buffer(5)]],
    device const float* beta[[buffer(6)]],
    constant uint& n[[buffer(7)]],
    constant uint& k[[buffer(8)]],
    constant float& softlog_alpha[[buffer(9)]],
    constant float& inhib_alpha[[buffer(10)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<k && i.y<n){
        uint idx=i.y*k+i.x;
        float X=X_g[idx]+X_m[idx]+E_t[idx];
        float aX=fabs(X);
        H_t1[idx]=((X)*(log(1.0f+softlog_alpha*aX)/(1e-20+aX))-(inhib_alpha*mu[i.y]))/(1e-20+beta[i.y]*gamma[i.y]);
    }
}