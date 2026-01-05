import Metal
import Foundation
import Darwin
@inline(__always)
func clamp01(_ x: Float) -> Float {
    return min(1.0, max(0.0, x))
}
public func hippocampus_step(
    stream: ComputeStream,
    _ HC_t: GPUBuffer<Float>,
    _ HC_p: GPUBuffer<Float>,
    _ HC_prev: GPUBuffer<Float>,
    _ H_t: GPUBuffer<Float>,
    _ H_p: GPUBuffer<Float>,
    _ P: GPUBuffer<Float>,
    _ Q: GPUBuffer<Float>,
    _ n_: UInt32,
    _ k_: UInt32,
    _ hr_: UInt32,
    _ epsilon: Float,
    _ alpha: Float,
    _ theta: Float,
    _ lambda: Float,
    _ DA_c: Float,
    _ ACh_c: Float,
    _ NE_c: Float
){
    let ACh0 = clamp01(ACh_c)
    let DA0  = clamp01(DA_c)
    let NE0  = clamp01(NE_c)
    copy(stream: stream, HC_t, HC_prev, n_*hr_, 1)
    gemm(stream: stream, H_t, P, H_p, n_, k_, hr_, 1)
    relu_s(stream: stream, H_p, H_p, n_*hr_, 1, theta*(1.0-0.3*NE0))
    add_scaled(stream: stream, HC_t, H_p, HC_t, n_*hr_, 1, 1.0-lambda, alpha*ACh0*(0.25+0.75*DA0)*(0.5+0.5*NE0))
    gemm(stream: stream, HC_prev, Q, HC_p, n_, hr_, k_, 1)
    add_scaled(stream: stream, H_t, HC_p, H_t, n_*k_, 1, 1, epsilon*(1.0-ACh0)*(0.5+0.5*DA0))
}
public final class HippocampusPrime{
    //gpu
    let device: MTLDevice
    let stream: ComputeStream
    //main
    var HC_t: GPUBuffer<Float>
    var P: GPUBuffer<Float>
    var Q: GPUBuffer<Float>
    //scratch
    var HC_p: GPUBuffer<Float>
    var HC_prev: GPUBuffer<Float>
    var H_p: GPUBuffer<Float>
    //params
    var n: UInt32
    var k: UInt32
    var hr: UInt32
    var epsilon: Float
    var alpha: Float
    var theta: Float
    var lambda: Float
    init(
        device: MTLDevice,
        stream: ComputeStream,
        n: UInt32,
        k: UInt32,
        hr: UInt32,
        P: GPUBuffer<Float>,
        Q: GPUBuffer<Float>,
        epsilon: Float,
        alpha: Float,
        theta: Float,
        lambda: Float
    ){
        self.device=device
        self.stream = stream
        self.HC_t=GPUBuffer<Float>(device: device, capacity: Int(n*hr))
        self.P=P
        self.Q=Q
        self.HC_p=GPUBuffer<Float>(device: device, capacity: Int(n*hr))
        self.HC_prev=GPUBuffer<Float>(device: device, capacity: Int(n*hr))
        self.H_p=GPUBuffer<Float>(device: device, capacity: Int(n*hr))
        self.n=n
        self.k=k
        self.hr=hr
        self.epsilon=epsilon
        self.alpha=alpha
        self.theta=theta
        self.lambda=lambda
    }
    func step(
        H_t: GPUBuffer<Float>,
        DA_c: Float,
        ACh_c: Float,
        NE_c: Float,
    ){
        hippocampus_step(
            stream: stream, 
            HC_t, 
            HC_p,
            HC_prev,
            H_t,
            H_p,
            P,
            Q,
            n,
            k,
            hr,
            epsilon,
            alpha,
            theta,
            lambda,
            DA_c,
            ACh_c,
            NE_c
        )
    }
    func zero_state(){
        zero(stream: stream, HC_t, n*hr, 1)
    }
}
