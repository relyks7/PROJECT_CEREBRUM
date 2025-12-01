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
struct Scalar {
    float* ptr;
    Scalar(float x) {
        ptr = new float[1]{x};
    }
    ~Scalar() {
        delete[] ptr;
    }
    float* data() const {return ptr;}
};
void op_add(float* A, float* B, float* C, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[5]={A, B, C, nval.data(), bval.data()};
    int32_t lens[5]={b*n, b*n, b*n, 1, 1};
    kernel_runner_call(
        "add",
        ptrs, lens, 5,
        n,b,1,
        1,1,1
    );
}
void op_div(float* A, float* B, float* C, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[5]={A, B, C, nval.data(), bval.data()};
    int32_t lens[5]={b*n, b*n, b*n, 1, 1};
    kernel_runner_call(
        "div",
        ptrs, lens, 5,
        n,b,1,
        1,1,1
    );
}
void op_mul(float* A, float* B, float* C, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[5]={A, B, C, nval.data(), bval.data()};
    int32_t lens[5]={b*n, b*n, b*n, 1, 1};
    kernel_runner_call(
        "mul",
        ptrs, lens, 5,
        n,b,1,
        1,1,1
    );
}
void op_sub(float* A, float* B, float* C, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[5]={A, B, C, nval.data(), bval.data()};
    int32_t lens[5]={b*n, b*n, b*n, 1, 1};
    kernel_runner_call(
        "sub",
        ptrs, lens, 5,
        n,b,1,
        1,1,1
    );
}
void op_embedding(float* A, int* B, float* C, int n, int d, int vocab_size){
    Scalar nval(n);
    Scalar dval(d);
    float* ptrs[5]={A, (float*)B, C, nval.data(), dval.data()};
    int32_t lens[5]={vocab_size*d, n, d*n, 1, 1};
    kernel_runner_call(
        "embedding",
        ptrs, lens, 5,
        n, 1, 1,
        1, 1, 1
    );
}
void op_gemm1(float* A, float* B, float* C, int m, int n, int p, int b){
    Scalar mval(m);
    Scalar nval(n);
    Scalar pval(p);
    Scalar bval(b);
    float* ptrs[7]={A, B, C, mval.data(), nval.data(), pval.data(), bval.data()};
    int32_t lens[7]={m*n*b, n*p*b, m*p*b, 1, 1, 1, 1};
    kernel_runner_call(
        "gemm1",
        ptrs, lens, 7,
        (p + 31) / 32, (m + 31) / 32, b,
        32, 32, 1
    );
}
void op_gemm2(float* A, float* B, float* C, int m, int n, int p, int b){
    Scalar mval(m);
    Scalar nval(n);
    Scalar pval(p);
    Scalar bval(b);
    float* ptrs[7]={A, B, C, mval.data(), nval.data(), pval.data(), bval.data()};
    int32_t lens[7]={m*n*b, n*p*b, m*p*b, 1, 1, 1, 1};
    kernel_runner_call(
        "gemm2",
        ptrs, lens, 7,
        (p + 31) / 32, (m + 31) / 32, b,
        32, 32, 1
    );
}
void op_gemm3(float* A, float* B, float* C, int m, int n, int p, int b){
    Scalar mval(m);
    Scalar nval(n);
    Scalar pval(p);
    Scalar bval(b);
    float* ptrs[7]={A, B, C, mval.data(), nval.data(), pval.data(), bval.data()};
    int32_t lens[7]={m*n*b, n*p*b, m*p*b, 1, 1, 1, 1};
    kernel_runner_call(
        "gemm3",
        ptrs, lens, 7,
        (p + 31) / 32, (m + 31) / 32, b,
        32, 32, 1
    );
}
void op_layernorm(float* A, float* B, float* mu, float* sigma2, int n, float eps, int b){
    Scalar nval(n);
    Scalar epsval(eps);
    Scalar bval(b);
    float* ptrs[7]={A, B, mu, sigma2, nval.data(), epsval.data(), bval.data()};
    int32_t lens[7]={b*n, b*n, b, b, 1, 1, 1};
    kernel_runner_call(
        "layernorm",
        ptrs, lens, 7,
        n, b, 1,
        1,1,1
    );
}
void op_max_simd(float* A, float* B, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[4]={A, B, nval.data(), bval.data()};
    int32_t lens[4]={b*n, b*(n+127)/128, 1, 1};
    kernel_runner_call(
        "max_simd_reduce",
        ptrs, lens, 4,
        (n+127)/128, b, 1, 
        128, 1, 1
    );
}
void op_sum_simd(float* A, float* B, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[4]={A, B, nval.data(), bval.data()};
    int32_t lens[4]={b*n, b*(n+127)/128, 1, 1};
    kernel_runner_call(
        "sum_simd_reduce",
        ptrs, lens, 4,
        (n+127)/128, b, 1, 
        128, 1, 1
    );
}
void op_mean_simd(float* A, float* B, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[4]={A, B, nval.data(), bval.data()};
    int32_t lens[4]={b*n, b*(n+127)/128, 1, 1};
    kernel_runner_call(
        "mean_simd_reduce",
        ptrs, lens, 4,
        (n+127)/128, b, 1, 
        128, 1, 1
    );
}
void op_softmax_simd(float* A, float* B, float* global_max, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[5]={A, B, global_max, nval.data(), bval.data()};
    int32_t lens[5]={b*n, b*(n+127)/128, b, 1, 1};
    kernel_runner_call(
        "softmax_simd_reduce",
        ptrs, lens, 5,
        (n+127)/128, b, 1, 
        128, 1, 1
    );
}
void op_variance_simd(float* A, float* B, float* mu, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[5]={A, B, mu, nval.data(), bval.data()};
    int32_t lens[5]={b*n, b*(n+127)/128, b, 1, 1};
    kernel_runner_call(
        "variance_simd_reduce",
        ptrs, lens, 5,
        (n+127)/128, b, 1, 
        128, 1, 1
    );
}
void op_tanh(float* A, float* B, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[4]={A, B, nval.data(), bval.data()};
    int32_t lens[4]={b*n, b*n, 1, 1};
    kernel_runner_call(
        "tanh",
        ptrs, lens, 4,
        n, b, 1,
        1, 1, 1
    );
}
void op_relu(float* A, float* B, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[4]={A, B, nval.data(), bval.data()};
    int32_t lens[4]={b*n, b*n, 1, 1};
    kernel_runner_call(
        "relu",
        ptrs, lens, 4,
        n, b, 1,
        1, 1, 1
    );
}
void op_softmax(float* A, float* B, float* global_max, float* denom, int n, int b){
    Scalar nval(n);
    Scalar bval(b);
    float* ptrs[6]={A, B, global_max, denom, nval.data(), bval.data()};
    int32_t lens[6]={b*n, b*n, b, b, 1, 1};
    kernel_runner_call(
        "softmax",
        ptrs, lens, 6,
        n, b, 1,
        1, 1, 1
    );
}
void op_outer_prod(float* A, float* B, float* C, int n, int m, int b){
    Scalar nval(n);
    Scalar mval(m);
    Scalar bval(b);
    op_gemm1(
        A, B, C, 
        n, 1, m,
        b
    );
}
int main(){
    kernel_runner_init();
    return 0;
}