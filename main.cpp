#include <iostream>
#include <vector>
#include <cstdint>
extern "C" {
    void kernel_runner_init();
    void kernel_runner_call(
        const char* kernelName,
        float** ptrArray,
        int32_t* lenArray,
        int32_t numBuffers,
        int32_t gridX, int32_t gridY, int32_t gridZ,
        int32_t tgX, int32_t tgY, int32_t tgZ
    );
}
void op_add(float* A, float* B, float* C, int n, int b){
    float nval=float(n);
    float bval=float(b);
    float* ptrs[5]={A, B, C, &nval, &bval};
    int32_t lens[5]={n, n, n, 1, 1};
    kernel_runner_call(
        "add",
        ptrs, lens, 5,
        n,b,1,
        1,1,1
    );
}
void op_div(float* A, float* B, float* C, int n, int b){
    float nval=float(n);
    float bval=float(b);
    float* ptrs[5]={A, B, C, &nval, &bval};
    int32_t lens[5]={n, n, n, 1, 1};
    kernel_runner_call(
        "div",
        ptrs, lens, 5,
        n,b,1,
        1,1,1
    );
}
void op_mul(float* A, float* B, float* C, int n, int b){
    float nval=float(n);
    float bval=float(b);
    float* ptrs[5]={A, B, C, &nval, &bval};
    int32_t lens[5]={n, n, n, 1, 1};
    kernel_runner_call(
        "mul",
        ptrs, lens, 5,
        n,b,1,
        1,1,1
    );
}
void op_sub(float* A, float* B, float* C, int n, int b){
    float nval=float(n);
    float bval=float(b);
    float* ptrs[5]={A, B, C, &nval, &bval};
    int32_t lens[5]={n, n, n, 1, 1};
    kernel_runner_call(
        "sub",
        ptrs, lens, 5,
        n,b,1,
        1,1,1
    );
}
