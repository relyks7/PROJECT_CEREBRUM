import Metal
import Foundation
import Darwin
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

    var lambda: Float
    var eta_max: Float
    var mes_alpha: Float
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
        w_DA: Float){
        self.n=n
        self.k=k
        self.r=r
        self.device=device
        self.stream=stream
        self.H_t0=GPUBuffer(device: device, capacity: Int(n*k))
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
    }
    func step(E_t: GPUBuffer<Float>){
        cortex_step(
            stream: stream,
            E_t,
            H_t0,
            H_t0_t,
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