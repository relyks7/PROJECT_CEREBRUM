import Metal
import Foundation
import Darwin
public func bg_step(
    stream: ComputeStream,
    _ W_str: GPUBuffer<Float>,
    _ b_str: GPUBuffer<Float>,
    _ H_t0: GPUBuffer<Float>,
    _ s0: GPUBuffer<Float>,
    _ s: GPUBuffer<Float>,
    _ g: GPUBuffer<Float>,
    _ M: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ ST: GPUBuffer<Float>,
    _ m_: UInt32,
    _ n_: UInt32,
    _ k_: UInt32,
    _ DA_c_: Float,
    _ alpha_: Float,
    _ beta_: Float,
    _ kappa_: Float,
    _ gamma_: Float
){
    gemv(stream: stream, W_str, H_t0, s0, m_, n_*k_)
    add(stream: stream, s0, b_str, s, m_, 1)
    bg_forward(stream: stream, s, g, M, ST, scratch0, scratch1, m_, DA_c_, alpha_, beta_, kappa_, gamma_)
}
public func bg_learn(
    stream: ComputeStream,
    _ g: GPUBuffer<Float>,
    _ H_t0: GPUBuffer<Float>,
    _ W_str: GPUBuffer<Float>,
    _ d_w: GPUBuffer<Float>,
    _ eta0: Float,
    _ eta_max: Float,
    _ w_max: Float,
    _ DA_c: Float,
    _ m: UInt32,
    _ n: UInt32,
    _ k: UInt32
){
    let eta = min(eta0 * max(0.0, DA_c), eta_max)
    outer_prod(stream: stream, g, H_t0, d_w, m, n*k, eta)
    add(stream: stream, W_str, d_w, W_str, m*n*k, 1)
    bg_oja(stream: stream, g, W_str, n*k, m, eta, w_max)
}
public final class BasalGangliaPrime{
    //gpu
    let device: MTLDevice
    let stream: ComputeStream
    //main
    var W_str: GPUBuffer<Float>
    var b_str: GPUBuffer<Float>
    var s: GPUBuffer<Float>
    var g: GPUBuffer<Float>
    //scratch
    var s0: GPUBuffer<Float>
    var d_w: GPUBuffer<Float>
    var scratch0: GPUBuffer<Float>
    var scratch1: GPUBuffer<Float>
    var ST: GPUBuffer<Float>
    var M: GPUBuffer<Float>
    //params
    var m: UInt32
    var n: UInt32
    var k: UInt32
    var alpha: Float
    var beta: Float
    var kappa: Float
    var gamma: Float
    var eta0: Float
    var eta_max: Float
    var w_max: Float
    init(
        device: MTLDevice,
        stream: ComputeStream,
        W_str: GPUBuffer<Float>,
        b_str: GPUBuffer<Float>,
        m: UInt32,
        n: UInt32,
        k: UInt32,
        alpha: Float,
        beta: Float,
        kappa: Float,
        eta0: Float,
        eta_max: Float,
        w_max: Float,
        gamma: Float
    ){
        self.device = device
        self.stream = stream
        self.W_str=W_str
        self.b_str=b_str
        self.m=m
        self.n=n
        self.k=k
        self.alpha=alpha
        self.beta=beta
        self.kappa=kappa
        self.s=GPUBuffer<Float>(device: device, capacity: Int(m))
        self.g=GPUBuffer<Float>(device: device, capacity: Int(m))
        self.d_w=GPUBuffer<Float>(device: device, capacity: Int(m*n*k))
        self.s0=GPUBuffer<Float>(device: device, capacity: Int(m))
        self.scratch0=GPUBuffer<Float>(device: device, capacity: Int(m))
        self.scratch1=GPUBuffer<Float>(device: device, capacity: Int(m))
        self.ST=GPUBuffer<Float>(device: device, capacity: 1)
        self.M=GPUBuffer<Float>(device: device, capacity: 1)
        self.eta0=eta0
        self.eta_max=eta_max
        self.w_max=w_max
        self.gamma=gamma
    }
    func step(
        H_t0: GPUBuffer<Float>,
        DA_c: Float
    ){
        bg_step(
            stream: stream,
            W_str,
            b_str,
            H_t0,
            s0,
            s,
            g,
            M,
            scratch0,
            scratch1,
            ST,
            m,
            n,
            k,
            DA_c,
            alpha,
            beta,
            kappa,
            gamma
        )
        stream.advance()
    }
    func learn(
        H_t0: GPUBuffer<Float>,
        DA_c: Float
    ){
        bg_learn(
            stream: stream,
            g,
            H_t0,
            W_str,
            d_w,
            eta0,
            eta_max,
            w_max,
            DA_c,
            m,
            n,
            k
        )
        stream.advance()
    }
}
public func bg_setup() -> BasalGangliaPrime{
    let W_str=GPUBuffer<Float>(device: device, capacity: Int(m*n*k))
    let b_str=GPUBuffer<Float>(device: device, capacity: Int(m))
    let wwscale: Float = 1.0 / sqrt(Float(n))
    for i in 0..<m*n*k {
        W_str.ptr()[Int(i)] = Float.random(in: -wwscale...wwscale)
    }
    for i in 0..<(m) {
        b_str.ptr()[Int(i)] = Float.random(in: (-1.0)...(-0.2))
    }
    stream.advance()
    stream.synchronize()
    let bg=BasalGangliaPrime(
        device: device,
        stream: stream,
        W_str: W_str,
        b_str: b_str,
        m: m,
        n: n, 
        k: k,
        alpha:1.0,
        beta:1.0,
        kappa:8.0,
        eta0: 1e-5,
        eta_max: 5e-5,
        w_max: wwscale,
        gamma: 1.00
    )
    return bg
}