#include <metal_stdlib>
using namespace metal;
kernel void get_eta(
    device const float* NE[[buffer(0)]],
    device const float* ACh[[buffer(1)]],
    device const float* DA[[buffer(2)]],
    device float* eta[[buffer(3)]],
    constant uint& n[[buffer(4)]],
    constant float& w_NE [[buffer(5)]],
    constant float& w_ACh [[buffer(6)]],
    constant float& w_DA [[buffer(7)]],
    constant float& b [[buffer(8)]],
    constant float& eta_max [[buffer(9)]],
    uint i[[thread_position_in_grid]]
){
    if (i<n){
        float z=w_NE*NE[i]+w_ACh*ACh[i]+w_DA*DA[i]+b;
        eta[i]=eta_max*max(0.0f, z * rsqrt(1.0f + z*z));
    }
}