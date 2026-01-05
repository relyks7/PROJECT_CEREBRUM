#include <metal_stdlib>
using namespace metal;
kernel void get_alpha(
    device const float* E_tm[[buffer(0)]],
    constant float& c[[buffer(1)]],
    float alpha [[buffer(2)]],
    uint2 i[[thread_position_in_grid]]
){
    float nv=E_tm[0];
    alpha=nv/(nv+c);
}