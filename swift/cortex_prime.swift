import Metal
import Foundation
import Darwin
public func cortex_step(
    stream: ComputeStream,
    _ E_t: GPUBuffer <Float>,
    _ H_t0: GPUBuffer <Float>,
    _ H_t0_t: GPUBuffer <Float>,
    _ H_scratch: GPUBuffer <Float>,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ B_t: GPUBuffer <Float>,
    _ X_g0: GPUBuffer <Float>,
    _ X_g: GPUBuffer <Float>,
    _ X_m3_t: GPUBuffer <Float>,
    _ X_m5_t: GPUBuffer <Float>,
    _ X_m7_t: GPUBuffer <Float>,
    _ X_m11_t: GPUBuffer <Float>,
    _ X_m3: GPUBuffer <Float>,
    _ X_m5: GPUBuffer <Float>,
    _ X_m7: GPUBuffer <Float>,
    _ X_m11: GPUBuffer <Float>,
    _ X_m: GPUBuffer <Float>,
    _ mu0: GPUBuffer <Float>,
    _ mu1: GPUBuffer <Float>,
    _ mu2: GPUBuffer <Float>,
    _ mu: GPUBuffer <Float>,
    _ gamma0: GPUBuffer <Float>,
    _ gamma1: GPUBuffer <Float>,
    _ gamma2: GPUBuffer <Float>,
    _ gamma: GPUBuffer <Float>,
    _ H_t1: GPUBuffer <Float>,
    _ W_conv_r3: GPUBuffer <Float>,
    _ W_conv_r5: GPUBuffer <Float>,
    _ W_conv_r7: GPUBuffer <Float>,
    _ W_conv_r11: GPUBuffer <Float>,
    _ W_inhib_sub_r3: GPUBuffer <Float>,
    _ W_inhib_div_r7: GPUBuffer <Float>,
    _ beta: GPUBuffer <Float>,
    _ softlog_alpha_: Float,
    _ mes_alpha: Float,
    _ inhib_alpha_: Float,
    _ alpha_gamma_: Float,
    _ dt: Float,
    _ leak_: Float,
    _ NE_c: Float,
    _ ACh_c: Float,
    _ DA_c: Float,
    _ w2_ACh_: Float,
    _ w2_NE_: Float,
    _ n_: UInt32,
    _ k_: UInt32,
    _ r_: UInt32
){
    gemm(stream: stream, B_t, H_t0, X_g0, r_, n_, k_, 1);
    gemm(stream: stream, A, X_g0, X_g, n_, r_, k_, 1);
    transpose(stream: stream, H_t0, H_t0_t, k_, n_)
    conv_r3(stream: stream, H_t0_t, W_conv_r3, X_m3_t, n_, k_);
    conv_r5(stream: stream, H_t0_t, W_conv_r5, X_m5_t, n_, k_);
    conv_r7(stream: stream, H_t0_t, W_conv_r7, X_m7_t, n_, k_);
    conv_r11(stream: stream, H_t0_t, W_conv_r11, X_m11_t, n_, k_);
    transpose(stream: stream, X_m3_t, X_m3, n_, k_)
    transpose(stream: stream, X_m5_t, X_m5, n_, k_)
    transpose(stream: stream, X_m7_t, X_m7, n_, k_)
    transpose(stream: stream, X_m11_t, X_m11, n_, k_)
    add4(stream: stream, X_m3, X_m5, X_m7, X_m11, X_m, n_*k_, 1, mes_alpha);
    inhib_sub_r3(stream: stream, H_t0, W_inhib_sub_r3, mu0, mu, mu1, mu2, k_, n_);
    inhib_div_r7(stream: stream, H_t0, W_inhib_div_r7, gamma0, gamma, gamma1, gamma2, k_, n_);
    var n=n_
    var k=k_
    var softlog_alpha=softlog_alpha_
    var inhib_alpha=inhib_alpha_
    var alpha_gamma=alpha_gamma_
    let leak=leak_
    var g0=1.0+w2_NE_*NE_c
    var g1=1.0-w2_ACh_*ACh_c
    stream.dispatch(
        kernel: "cortex_step",
        args: [
            .buffer(E_t.buffer),
            .buffer(H_t1.buffer),
            .buffer(X_g.buffer),
            .buffer(X_m.buffer),
            .buffer(mu.buffer),
            .buffer(gamma.buffer),
            .buffer(beta.buffer),
            bytes(&n),
            bytes(&k),
            bytes(&softlog_alpha),
            bytes(&inhib_alpha),
            bytes(&alpha_gamma),
            bytes(&g0),
            bytes(&g1)
        ],
        grid: MTLSize(
            width: (Int(k)+255)/256,
            height: Int(n),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
    axbpy(stream: stream, H_t0, H_t1, H_scratch, n_*k_, 1, dt, leak*exp(-DA_c))
    copy(stream:stream, H_scratch, H_t0, n_*k_, 1)
}
//NB: ASSUME B_t IS ALREADY SET (given that B is constant)
public func oja(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ A_new: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ eta: GPUBuffer<Float>,
    _ B_t: GPUBuffer<Float>,
    _ T1: GPUBuffer<Float>,
    _ T2: GPUBuffer<Float>,
    _ S: GPUBuffer<Float>,
    _ U: GPUBuffer<Float>,
    _ G: GPUBuffer<Float>,
    _ H: GPUBuffer<Float>,
    _ X: GPUBuffer<Float>,
    _ Y: GPUBuffer<Float>,
    _ Y_t: GPUBuffer<Float>,
    _ r_: UInt32,
    _ n_: UInt32,
    _ k_: UInt32,
    _ lambda_: Float
){
    transpose(stream: stream, Y, Y_t, k_, n_)
    gemm(stream: stream, Y_t, B, T1, k_, n_, r_, 1)
    gemm(stream: stream, Y_t, A, T2, k_, n_, r_, 1)
    gemm(stream: stream, B_t, B, S, r_, n_, r_, 1)
    gemm(stream: stream, T2, S, U, k_, r_, r_, 1)
    gemm(stream: stream, X, T1, G, n_, k_, r_, 1)
    gemm(stream: stream, Y, U, H, n_, k_, r_, 1)
    final_oja_step(stream: stream, G, H, eta, A, A_new, n_, r_, lambda_)
    copy(stream: stream, A_new, A, n_*r_, 1)
}
public final class CortexPrime{
    //gpu
    let device: MTLDevice
    let stream: ComputeStream
    //main
    var H_t0: GPUBuffer <Float>
    var H_t1: GPUBuffer <Float>

    var A: GPUBuffer <Float>
    var A_new: GPUBuffer<Float>

    var B: GPUBuffer <Float>
    var B_t: GPUBuffer<Float>

    var X_g: GPUBuffer <Float>
    var X_m: GPUBuffer <Float>
    var mu: GPUBuffer <Float>
    var gamma: GPUBuffer <Float>

    var eta: GPUBuffer<Float>
    var H_t1_t: GPUBuffer<Float>
    //scratch
    var H_t0_t: GPUBuffer<Float>
    var H_scratch: GPUBuffer<Float>
    var X_g0: GPUBuffer <Float>
    var X_m3_t: GPUBuffer <Float>
    var X_m5_t: GPUBuffer <Float>
    var X_m7_t: GPUBuffer <Float>
    var X_m11_t: GPUBuffer <Float>
    var X_m3: GPUBuffer <Float>
    var X_m5: GPUBuffer <Float>
    var X_m7: GPUBuffer <Float>
    var X_m11: GPUBuffer <Float>
    var mu0: GPUBuffer <Float>
    var mu1: GPUBuffer <Float>
    var mu2: GPUBuffer <Float>

    var gamma0: GPUBuffer <Float>
    var gamma1: GPUBuffer <Float>
    var gamma2: GPUBuffer <Float>

    var T1: GPUBuffer<Float>
    var T2: GPUBuffer<Float>
    var S: GPUBuffer<Float>
    var U: GPUBuffer<Float>
    var G: GPUBuffer<Float>
    var H: GPUBuffer<Float>
    //params
    let n: UInt32
    let k: UInt32
    let r: UInt32

    let w_NE: Float
    let w_ACh: Float
    let w_DA: Float

    var W_conv_r3: GPUBuffer <Float>
    var W_conv_r5: GPUBuffer <Float>
    var W_conv_r7: GPUBuffer <Float>
    var W_conv_r11: GPUBuffer <Float>

    var W_inhib_sub_r3: GPUBuffer <Float>
    var W_inhib_div_r7: GPUBuffer <Float>

    var beta: GPUBuffer <Float>

    var oja_bias: Float

    var softlog_alpha: Float
    var inhib_alpha: Float
    var alpha_gamma: Float
    var lambda: Float
    var eta_max: Float
    var mes_alpha: Float
    var dt: Float
    var leak: Float
    var w2_NE: Float
    var w2_ACh: Float
    init(device: MTLDevice, 
        stream: ComputeStream, 
        n: UInt32, 
        k: UInt32, 
        r: UInt32,
        B: GPUBuffer<Float>,
        W_conv_r3: GPUBuffer<Float>,
        W_conv_r5: GPUBuffer<Float>,
        W_conv_r7: GPUBuffer<Float>,
        W_conv_r11: GPUBuffer<Float>,
        W_inhib_sub_r3: GPUBuffer<Float>,
        W_inhib_div_r7: GPUBuffer<Float>,
        beta: GPUBuffer<Float>,
        softlog_alpha: Float,
        mes_alpha: Float,
        inhib_alpha: Float,
        lambda: Float,
        eta_max: Float,
        oja_bias: Float,
        w_NE: Float, 
        w_ACh: Float, 
        w_DA: Float,
        w2_NE: Float,
        w2_ACh: Float,
        dt: Float,
        alpha_gamma: Float,
        leak: Float){
        self.n=n
        self.k=k
        self.r=r
        self.device=device
        self.stream=stream
        self.H_t0=GPUBuffer(device: device, capacity: Int(n*k))
        self.H_scratch=GPUBuffer(device: device, capacity: Int(n*k))
        self.H_t1=GPUBuffer(device: device, capacity: Int(n*k))
        self.A=GPUBuffer(device: device, capacity: Int(n*r))
        self.A_new=GPUBuffer(device: device, capacity: Int(n*r))
        self.B=GPUBuffer(device: device, capacity: Int(n*r))
        self.B_t=GPUBuffer(device: device, capacity: Int(r*n))
        copy(stream: stream, B, self.B, n*r, 1)
        transpose(stream: stream, B, self.B_t, n, r)
        self.X_g=GPUBuffer(device: device, capacity: Int(n*k))
        self.X_m=GPUBuffer(device: device, capacity: Int(n*k))
        self.mu=GPUBuffer(device: device, capacity: Int(n))
        self.gamma=GPUBuffer(device: device, capacity: Int(n))
        self.eta=GPUBuffer(device: device, capacity: Int(n))
        self.H_t0_t=GPUBuffer(device: device, capacity: Int(k*n))
        self.H_t1_t=GPUBuffer(device: device, capacity: Int(k*n))
        self.X_g0=GPUBuffer(device: device, capacity: Int(r*k))
        self.X_m3_t=GPUBuffer(device: device, capacity: Int(k*n))
        self.X_m5_t=GPUBuffer(device: device, capacity: Int(k*n))
        self.X_m7_t=GPUBuffer(device: device, capacity: Int(k*n))
        self.X_m11_t=GPUBuffer(device: device, capacity: Int(k*n))
        self.X_m3=GPUBuffer(device: device, capacity: Int(n*k))
        self.X_m5=GPUBuffer(device: device, capacity: Int(n*k))
        self.X_m7=GPUBuffer(device: device, capacity: Int(n*k))
        self.X_m11=GPUBuffer(device: device, capacity: Int(n*k))
        self.mu0=GPUBuffer(device: device, capacity: Int(n*k))
        self.mu1=GPUBuffer(device: device, capacity: Int(n*k))
        self.mu2=GPUBuffer(device: device, capacity: Int(n*k))
        self.gamma0=GPUBuffer(device: device, capacity: Int(n*k))
        self.gamma1=GPUBuffer(device: device, capacity: Int(n*k))
        self.gamma2=GPUBuffer(device: device, capacity: Int(n*k))
        self.T1=GPUBuffer(device: device, capacity: Int(k*r))
        self.T2=GPUBuffer(device: device, capacity: Int(k*r))
        self.S=GPUBuffer(device: device, capacity: Int(r*r))
        self.U=GPUBuffer(device: device, capacity: Int(k*r))
        self.G=GPUBuffer(device: device, capacity: Int(n*r))
        self.H=GPUBuffer(device: device, capacity: Int(n*r))
        self.W_conv_r3=W_conv_r3
        self.W_conv_r5=W_conv_r5
        self.W_conv_r7=W_conv_r7
        self.W_conv_r11=W_conv_r11
        self.W_inhib_sub_r3=W_inhib_sub_r3
        self.W_inhib_div_r7=W_inhib_div_r7
        self.beta=beta
        self.softlog_alpha=softlog_alpha
        self.inhib_alpha=inhib_alpha
        self.lambda=lambda
        self.eta_max=eta_max
        self.oja_bias=oja_bias
        self.w_NE=w_NE
        self.w_ACh=w_ACh
        self.w_DA=w_DA
        self.mes_alpha=mes_alpha
        self.dt=dt
        self.alpha_gamma=alpha_gamma
        self.leak=leak
        self.w2_NE=w2_NE
        self.w2_ACh=w2_ACh
    }
    func step(
        E_t: GPUBuffer<Float>,
        NE_c: Float,
        ACh_c: Float,
        DA_c: Float
    ){
        cortex_step(
            stream: stream,
            E_t,
            H_t0,
            H_t0_t,
            H_scratch,
            A,
            B, 
            B_t,
            X_g0,
            X_g,
            X_m3_t,
            X_m5_t,
            X_m7_t,
            X_m11_t,
            X_m3,
            X_m5,
            X_m7,
            X_m11,
            X_m,
            mu0,
            mu1,
            mu2,
            mu,
            gamma0,
            gamma1,
            gamma2,
            gamma,
            H_t1,
            W_conv_r3,
            W_conv_r5,
            W_conv_r7,
            W_conv_r11,
            W_inhib_sub_r3,
            W_inhib_div_r7,
            beta,
            softlog_alpha,
            mes_alpha,
            inhib_alpha,
            alpha_gamma,
            dt,
            leak,
            NE_c,
            ACh_c,
            DA_c,
            w2_ACh,
            w2_NE,
            n,
            k,
            r
        )
        
        stream.advance()
    }
    func learn(
        NE: GPUBuffer <Float>,
        ACh: GPUBuffer <Float>,
        DA: GPUBuffer <Float>
    ){
        get_eta(
            stream: stream,
            NE,
            ACh,
            DA,
            eta,
            n,
            w_NE,
            w_ACh,
            w_DA,
            oja_bias,
            eta_max
        )
        oja(
            stream: stream,
            A,
            A_new,
            B,
            eta,
            B_t,
            T1,
            T2,
            S,
            U,
            G,
            H,
            H_t0,
            H_t1,
            H_t1_t,
            r,
            n,
            k,
            lambda
        )
        stream.advance()
    }
    func zero_state(){
        zero(stream: stream, self.H_t0, n*k, 1)
        zero(stream: stream, self.H_t1, n*k, 1)
        zero(stream: stream, mu, n, 1)
        zero(stream: stream, gamma, n, 1)
        stream.advance()
    }
}
public func cortex_setup() -> CortexPrime{
    let B = GPUBuffer<Float>(device: device, capacity: Int(n*r))

    let W_conv_r3  = GPUBuffer<Float>(device: device, capacity: 7)
    let W_conv_r5  = GPUBuffer<Float>(device: device, capacity: 11)
    let W_conv_r7  = GPUBuffer<Float>(device: device, capacity: 15)
    let W_conv_r11 = GPUBuffer<Float>(device: device, capacity: 23)

    let W_inhib_sub_r3 = GPUBuffer<Float>(device: device, capacity: 7)
    let W_inhib_div_r7 = GPUBuffer<Float>(device: device, capacity: 15)

    let beta   = GPUBuffer<Float>(device: device, capacity: Int(n))
    zero(stream: stream, beta, n, 1)
    zero(stream: stream, W_conv_r3, 7, 1)
    zero(stream: stream, W_conv_r5, 11, 1)
    zero(stream: stream, W_conv_r7, 15, 1)
    zero(stream: stream, W_conv_r11, 23, 1)
    zero(stream: stream, W_inhib_sub_r3, 7, 1)
    zero(stream: stream, W_inhib_div_r7, 15, 1)
    stream.advance()
    stream.synchronize()
    for i in 0..<Int(n) {
        beta.ptr()[i] = 1.5
    }
    let a_conv_r3: [Float] = [
        0.05, 0.10, 0.20, 0.30, 0.20, 0.10, 0.05
    ]
    let a_conv_r5: [Float] = [
        0.02, 0.04, 0.08, 0.12, 0.16,
        0.20,
        0.16, 0.12, 0.08, 0.04, 0.02
    ]
    let a_conv_r7: [Float] = [
        0.01, 0.02, 0.04, 0.06, 0.09,
        0.12, 0.15, 0.18, 0.15, 0.12,
        0.09, 0.06, 0.04, 0.02, 0.01
    ]
    let a_conv_r11: [Float] = [
        0.004, 0.008, 0.015, 0.025, 0.040,
        0.060, 0.080, 0.100, 0.120, 0.140,
        0.160, 0.180, 0.160, 0.140, 0.120,
        0.100, 0.080, 0.060, 0.040, 0.025,
        0.015, 0.008, 0.004
    ]
    let a_inhib_sub_r3: [Float] = [
        0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25
    ]
    let a_inhib_div_r7: [Float] = [
        0.05, 0.08, 0.12, 0.18, 0.25,
        0.32, 0.38, 0.42, 0.38, 0.32,
        0.25, 0.18, 0.12, 0.08, 0.05
    ]
    load(a_conv_r3, into: W_conv_r3)
    load(a_conv_r5, into: W_conv_r5)
    load(a_conv_r7, into: W_conv_r7)
    load(a_conv_r11, into: W_conv_r11)
    load(a_inhib_sub_r3, into: W_inhib_sub_r3)
    load(a_inhib_div_r7, into: W_inhib_div_r7)
    for i in 0..<(n*r) {
        B.ptr()[Int(i)] = Float.random(in: -0.005...0.005)
    }
    let scaleB = 1.0 / sqrt(Float(n))
    for i in 0..<(n * r) {
        B.ptr()[Int(i)] *= scaleB
    }

    let cortex = CortexPrime(
        device: device,
        stream: stream,
        n: n,
        k: k,
        r: r,
        B: B,
        W_conv_r3: W_conv_r3,
        W_conv_r5: W_conv_r5,
        W_conv_r7: W_conv_r7,
        W_conv_r11: W_conv_r11,
        W_inhib_sub_r3: W_inhib_sub_r3,
        W_inhib_div_r7: W_inhib_div_r7,
        beta: beta,
        softlog_alpha: 0.8,
        mes_alpha: 0.08,
        inhib_alpha: 1.2,
        lambda: 1e-4,
        eta_max: 1e-2,
        oja_bias: 0.0,
        w_NE: 0.5,
        w_ACh: 1.0,
        w_DA: 2.0,
        w2_NE:0.5,
        w2_ACh:0.6,
        dt: 0.02,
        alpha_gamma: 1.0,
        leak: 0.003
    )
    stream.advance()
    stream.synchronize()
    for i in 0..<(n*r) {
        cortex.A.ptr()[Int(i)] = Float.random(in: -0.01...0.01)
    }
    let scaleA = 1.0 / sqrt(Float(r))
    for i in 0..<(n * r) {
        cortex.A.ptr()[Int(i)] *= scaleA
    }
    for i in 0..<Int(n * k) {
        cortex.H_t0.ptr()[i] = Float.random(in: -0.01...0.01)
    }
    return cortex
}