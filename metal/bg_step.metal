#include <metal_stdlib>
using namespace metal;
kernel void bg_step(
    device const float* S[[buffer(0)]],
    device float* G[[buffer(1)]],
    constant uint& n[[buffer(2)]],
    constant float& DA_c[[buffer(3)]],
    constant float& alpha[[buffer(4)]],
    constant float& beta[[buffer(5)]],
    constant float& kappa[[buffer(6)]],
    uint idx[[thread_position_in_grid]]
){
    if (idx<n){
        float da = clamp(DA_c, 0.0f, 1.0f);
        float go=S[idx]+alpha*da;
        float ngo=S[idx]-beta*da;
        go  = 1/(1+exp(-go));
        ngo = 1/(1+exp(-ngo));
        float U=(go-ngo)*kappa;
        G[idx]=U;
    }
}