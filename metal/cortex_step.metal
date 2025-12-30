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
    constant float& gamma0 [[buffer(11)]],
    constant float& g0 [[buffer(12)]],
    constant float& g1 [[buffer(13)]],
    constant float& g2 [[buffer(14)]],
    uint2 i[[thread_position_in_grid]]
){
    if (i.x < k && i.y < n) {
        uint idx = i.y * k + i.x;

        float X  = g0*(g1*(X_g[idx] + X_m[idx])+g2*E_t[idx]);
        float aX = fabs(X);

        float softlog_val = X * log(1.0f + softlog_alpha * aX) / (aX + 1e-6f);

        float div = max(gamma0 + beta[i.y] * gamma[i.y], 1e-3f);
        float sub = inhib_alpha * mu[i.y];

        H_t1[idx] = (softlog_val - sub) / div;
    }
}