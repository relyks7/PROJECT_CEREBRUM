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
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ k_: UInt32,
    _ hr_: UInt32,
    _ epsilon: Float,
    _ alpha: Float,
    _ theta: Float,
    _ lambda: Float,
    _ phase: Float,
    _ tau_b: Float,
    _ DA_c: Float,
    _ ACh_c: Float,
    _ NE_c: Float
){
    let ACh0 = clamp01(ACh_c)
    let DA0  = clamp01(DA_c)
    let NE0  = clamp01(NE_c)
    copy(stream: stream, HC_t, HC_prev, n_*hr_, 1)
    gemm(stream: stream, H_t, P, H_p, n_, k_, hr_, 1)
    add_scaled(stream: stream, H_p, B, H_p, n_*hr_, 1, 1, sin(phase))
    relu_s(stream: stream, H_p, H_p, n_*hr_, 1, theta*(1.0-0.3*NE0)*(1+sin(phase)*tau_b))
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
    var B: GPUBuffer<Float>
    var phase: Float=0
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
    var tau_b: Float
    let omega: Float=2*3.141592653589793/80
    init(
        device: MTLDevice,
        stream: ComputeStream,
        n: UInt32,
        k: UInt32,
        hr: UInt32,
        P: GPUBuffer<Float>,
        Q: GPUBuffer<Float>,
        B: GPUBuffer<Float>,
        epsilon: Float,
        alpha: Float,
        theta: Float,
        lambda: Float,
        tau_b: Float
    ){
        self.device=device
        self.stream = stream
        self.HC_t=GPUBuffer<Float>(device: device, capacity: Int(n*hr))
        self.P=P
        self.Q=Q
        self.B=B
        self.HC_p=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.HC_prev=GPUBuffer<Float>(device: device, capacity: Int(n*hr))
        self.H_p=GPUBuffer<Float>(device: device, capacity: Int(n*hr))
        self.n=n
        self.k=k
        self.hr=hr
        self.epsilon=epsilon
        self.alpha=alpha
        self.theta=theta
        self.lambda=lambda
        self.tau_b=tau_b
    }
    func step(
        H_t: GPUBuffer<Float>,
        DA_c: Float,
        ACh_c: Float,
        NE_c: Float
    ){
        phase += omega
        if phase > 2 * 3.141592653589793 { phase -= 2 * 3.141592653589793 }
        hippocampus_step(
            stream: stream, 
            HC_t, 
            HC_p,
            HC_prev,
            H_t,
            H_p,
            P,
            Q,
            B,
            n,
            k,
            hr,
            epsilon,
            alpha,
            theta,
            lambda,
            phase,
            tau_b,
            DA_c,
            ACh_c,
            NE_c
        )
        stream.advance()
    }
    func zero_state(){
        zero(stream: stream, HC_t, n*hr, 1)
    }
}
public func hippocampus_setup() -> HippocampusPrime{
    let P=GPUBuffer<Float>(device: device, capacity: Int(k*hr))
    let Q=GPUBuffer<Float>(device: device, capacity: Int(hr*k))
    let B=GPUBuffer<Float>(device: device, capacity: Int(n*hr))
    let pScale: Float = 1.0 / sqrt(Float(k))
    let noiseScale: Float = 0.02 * pScale
    for i in 0..<Int(k*hr) {
        P.ptr()[i] = Float.random(in: -pScale...pScale)
    }
    for i in 0..<Int(n*hr) {
        B.ptr()[i] = Float.random(in: -pScale...pScale)
    }
    for i in 0..<hr {
        for j in 0..<k {
            let noise = Float.random(in: -noiseScale...noiseScale)
            Q.ptr()[Int(i*k+j)] = P.ptr()[Int(j*hr+i)] + noise
        }
    }
    stream.advance()
    stream.synchronize()
    let hc=HippocampusPrime(
        device: device,
        stream: stream,
        n: n,
        k: k,
        hr: hr,
        P: P,
        Q: Q,
        B: B,
        epsilon: 0.02,
        alpha: 0.05,
        theta: 0.2,
        lambda: 0.01,
        tau_b: 0.15
    )
    return hc
}