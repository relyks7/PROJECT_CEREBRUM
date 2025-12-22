#include <metal_stdlib>
using namespace metal;
kernel void cortex_step(
    device float* H_t1[[buffer(0)]],
    device const float* X_g[[buffer(1)]],
    device const float* X_m[[buffer(2)]],
    device const float* mu[[buffer(3)]],
    device const float* gamma[[buffer(4)]],
    device const float* beta[[buffer(5)]],
    constant uint& n[[buffer(6)]],
    constant uint& k[[buffer(7)]],
    constant float& softlog_alpha[[buffer(8)]],
    constant float& inhib_alpha[[buffer(9)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x<k && i.y<n){
        uint idx=i.y*k+i.x;
        float X=X_g[idx]+X_m[idx];
        float aX=fabs(X);
        H_t1[idx]=((X)*(log(1.0f+softlog_alpha*aX)/(1e-20+aX))-(inhib_alpha*mu[i.y]))/(1e-20+beta[i.y]*gamma[i.y]);
    }
}