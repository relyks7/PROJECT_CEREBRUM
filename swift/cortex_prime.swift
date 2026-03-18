import Metal
import Foundation
import Darwin
public func cortex_step(
    stream: ComputeStream,
    _ E_t: GPUBuffer <Float>,
    _ elig: GPUBuffer <Float>,
    _ l2_: Float, 
    _ H_t0: GPUBuffer <Float>,
    _ H_t0_t: GPUBuffer <Float>,
    _ H_scratch: GPUBuffer <Float>,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ W_pred: GPUBuffer <Float>,
    _ W_act: GPUBuffer <Float>,
    _ a_t: GPUBuffer <Float>,
    _ U_t1: GPUBuffer<Float>,
    _ B_t: GPUBuffer <Float>,
    _ X_g0: GPUBuffer <Float>,
    _ X_g1: GPUBuffer <Float>,
    _ scratch0: GPUBuffer <Float>,
    _ scratch1: GPUBuffer <Float>,
    _ M: GPUBuffer <Float>,
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
    _ ACh_c_: Float,
    _ DA_c: Float,
    _ gw_DA: Float,
    _ w2_NE_: Float,
    _ n_: UInt32,
    _ k_: UInt32,
    _ r_: UInt32,
    _ Ds_: UInt32,
    _ ad_: UInt32
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
    var ACh_c=ACh_c_
    let l2=l2_
    var g_DA=gw_DA*DA_c
    add3(stream: stream, X_g, X_m, E_t, X_g1, n_*k_, 1)
    abs_mean_simd(stream: stream, X_g1, scratch0, scratch1, M, n_*k_, 1)
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
            .buffer(M.buffer),
            bytes(&n),
            bytes(&k),
            bytes(&softlog_alpha),
            bytes(&inhib_alpha),
            bytes(&alpha_gamma),
            bytes(&ACh_c),
            bytes(&g_DA)
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
    axbpy(stream: stream, H_t0, H_t1, H_scratch, n_*k_, 1, dt * (1.0 + w2_NE_ * NE_c), leak * exp(-min(max(DA_c, -0.3), 0.3)))
    copy(stream:stream, H_scratch, H_t0, n_*k_, 1)
    axbpy(stream: stream, elig, H_t0, elig, n_*k_, 1, 1.0-l2, 0)
    gemv(stream:stream, W_pred, H_t0, U_t1, Ds_, n_*k_)
    gemv(stream:stream, W_act, H_t0, a_t, ad_, n_*k_)
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
    _ lambda_: Float,
    _ DA_c_: Float
){
    transpose(stream: stream, Y, Y_t, k_, n_)
    gemm(stream: stream, Y_t, B, T1, k_, n_, r_, 1)
    gemm(stream: stream, Y_t, A, T2, k_, n_, r_, 1)
    gemm(stream: stream, B_t, B, S, r_, n_, r_, 1)
    gemm(stream: stream, T2, S, U, k_, r_, r_, 1)
    gemm(stream: stream, X, T1, G, n_, k_, r_, 1)
    gemm(stream: stream, Y, U, H, n_, k_, r_, 1)
    final_oja_step(stream: stream, G, H, eta, A, A_new, n_, r_, lambda_, DA_c_)
    copy(stream: stream, A_new, A, n_*r_, 1)
}
public func update_chemicals_public(
    _ prediction_error: GPUBuffer<Float>,
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ err_sign: GPUBuffer<Float>,
    _ err_abs: GPUBuffer<Float>,
    _ err_sq: GPUBuffer<Float>,
    _ Ds: UInt32,
    _ DA_c: inout Float,
    _ ACh_c: inout Float,
    _ NE_c: inout Float,
    _ d_DA: Float,
    _ d_ACh:  Float,
    _ d_NE: Float,
    _ k_DA: Float,
    _ k_ACh: Float,
    _ k_NE: Float
){
    mean_simd(stream: stream, prediction_error, scratch0, scratch1, err_sign, Ds, 1)
    abs_mean_simd(stream: stream, prediction_error, scratch0, scratch1, err_abs, Ds, 1)
    sqmean_simd(stream: stream, prediction_error, scratch0, scratch1, err_sq, Ds, 1)
    stream.advance()
    stream.synchronize()
    DA_c = d_DA * DA_c + (1 - d_DA) * (1 - 2 * tanh(err_abs.ptr()[0] * k_DA))
    NE_c = d_NE * NE_c + (1 - d_NE) * tanh(err_abs.ptr()[0] * k_NE)
    ACh_c = d_ACh * ACh_c + (1 - d_ACh) * tanh(err_sq.ptr()[0] * k_ACh)
}
public final class CortexPrime{
    //gpu
    let device: MTLDevice
    let stream: ComputeStream
    //main
    var H_t0: GPUBuffer <Float>
    var H_t0_norm: GPUBuffer <Float>
    var H_t1: GPUBuffer <Float>
    var elig: GPUBuffer <Float>
    var eligmean: GPUBuffer <Float>
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
    var eta_pred: Float
    var prediction_error: GPUBuffer<Float>

    var err_sign: GPUBuffer<Float>
    var err_abs: GPUBuffer<Float>
    var err_sq: GPUBuffer<Float>
    //scratch
    var H_t0_t: GPUBuffer<Float>
    var H_scratch: GPUBuffer<Float>
    var X_g0: GPUBuffer <Float>
    var X_g1: GPUBuffer <Float>
    var scratch0: GPUBuffer <Float>
    var scratch1: GPUBuffer <Float>
    var scratch2: GPUBuffer <Float>
    var scratch3: GPUBuffer <Float>
    var M: GPUBuffer <Float>
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
    let Ds: UInt32
    let ad: UInt32
    let lambda: Float
    var W_pred: GPUBuffer <Float>
    var W_act: GPUBuffer <Float>
    var a_t: GPUBuffer <Float>
    var U_t1: GPUBuffer <Float>
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
    var alpha_eta: Float
    var l2: Float
    var gw_DA: Float
    var eta_max: Float
    var mes_alpha: Float
    var dt: Float
    var leak: Float
    var w2_NE: Float
    var DA_c: Float=0
    var ACh_c: Float=0
    var NE_c: Float=0
    var d_DA: Float
    var d_ACh:  Float
    var d_NE: Float
    var k_DA: Float
    var k_ACh: Float
    var k_NE: Float
    init(device: MTLDevice, 
        stream: ComputeStream, 
        n: UInt32, 
        k: UInt32, 
        r: UInt32,
        Ds: UInt32,
        ad: UInt32,
        B: GPUBuffer<Float>,
        W_pred: GPUBuffer<Float>,
        W_act: GPUBuffer<Float>,
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
        l2: Float,
        eta_max: Float,
        oja_bias: Float,
        w2_NE: Float,
        dt: Float,
        alpha_gamma: Float,
        leak: Float,
        gw_DA: Float,
        alpha_eta: Float,
        d_DA: Float,
        d_ACh:  Float,
        d_NE: Float,
        k_DA: Float,
        k_ACh: Float,
        k_NE: Float,
        eta_pred: Float){
        self.n=n
        self.k=k
        self.r=r
        self.Ds=Ds
        self.ad=ad
        self.device=device
        self.stream=stream
        self.prediction_error=GPUBuffer<Float>(device: device, capacity: Int(Ds))
        self.err_sign=GPUBuffer<Float>(device: device, capacity: 1)
        self.err_abs=GPUBuffer<Float>(device: device, capacity: 1)
        self.err_sq=GPUBuffer<Float>(device: device, capacity: 1)
        self.d_DA=d_DA
        self.d_ACh=d_ACh
        self.d_NE=d_NE
        self.k_DA=k_DA
        self.k_ACh=k_ACh
        self.k_NE=k_NE
        self.eta_pred=eta_pred
        self.H_t0=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.H_t0_norm=GPUBuffer<Float>(device: device, capacity: 1)
        self.H_scratch=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.H_t1=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.elig=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.eligmean=GPUBuffer<Float>(device: device, capacity: Int(n))
        self.A=GPUBuffer<Float>(device: device, capacity: Int(n*r))
        self.A_new=GPUBuffer<Float>(device: device, capacity: Int(n*r))
        self.B=GPUBuffer<Float>(device: device, capacity: Int(n*r))
        self.B_t=GPUBuffer<Float>(device: device, capacity: Int(r*n))
        copy(stream: stream, B, self.B, n*r, 1)
        transpose(stream: stream, B, self.B_t, n, r)
        self.X_g=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.X_m=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.mu=GPUBuffer<Float>(device: device, capacity: Int(n))
        self.gamma=GPUBuffer<Float>(device: device, capacity: Int(n))
        self.eta=GPUBuffer<Float>(device: device, capacity: Int(n))
        self.H_t0_t=GPUBuffer<Float>(device: device, capacity: Int(k*n))
        self.H_t1_t=GPUBuffer<Float>(device: device, capacity: Int(k*n))
        self.X_g0=GPUBuffer<Float>(device: device, capacity: Int(r*k))
        self.X_g1=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.scratch0=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.scratch1=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.scratch2=GPUBuffer<Float>(device: device, capacity: Int(Ds))
        self.scratch3=GPUBuffer<Float>(device: device, capacity: Int(Ds))
        self.M=GPUBuffer<Float>(device: device, capacity: 1)
        self.X_m3_t=GPUBuffer<Float>(device: device, capacity: Int(k*n))
        self.X_m5_t=GPUBuffer<Float>(device: device, capacity: Int(k*n))
        self.X_m7_t=GPUBuffer<Float>(device: device, capacity: Int(k*n))
        self.X_m11_t=GPUBuffer<Float>(device: device, capacity: Int(k*n))
        self.X_m3=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.X_m5=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.X_m7=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.X_m11=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.mu0=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.mu1=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.mu2=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.gamma0=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.gamma1=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.gamma2=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.T1=GPUBuffer<Float>(device: device, capacity: Int(k*r))
        self.T2=GPUBuffer<Float>(device: device, capacity: Int(k*r))
        self.S=GPUBuffer<Float>(device: device, capacity: Int(r*r))
        self.U=GPUBuffer<Float>(device: device, capacity: Int(k*r))
        self.G=GPUBuffer<Float>(device: device, capacity: Int(n*r))
        self.H=GPUBuffer<Float>(device: device, capacity: Int(n*r))
        self.U_t1=GPUBuffer<Float>(device: device, capacity: Int(Ds))
        self.W_pred=W_pred
        self.W_act=W_act
        self.a_t=GPUBuffer<Float>(device: device, capacity: Int(ad))
        self.W_conv_r3=W_conv_r3
        self.W_conv_r5=W_conv_r5
        self.W_conv_r7=W_conv_r7
        self.W_conv_r11=W_conv_r11
        self.W_inhib_sub_r3=W_inhib_sub_r3
        self.W_inhib_div_r7=W_inhib_div_r7
        self.beta=beta
        self.softlog_alpha=softlog_alpha
        self.inhib_alpha=inhib_alpha
        self.l2=l2
        self.lambda=lambda
        self.eta_max=eta_max
        self.oja_bias=oja_bias
        self.mes_alpha=mes_alpha
        self.dt=dt
        self.alpha_gamma=alpha_gamma
        self.leak=leak
        self.w2_NE=w2_NE
        self.gw_DA=gw_DA
        self.alpha_eta=alpha_eta
    }
    func step(
        E_t: GPUBuffer<Float>
    ){
        cortex_step(
            stream: stream,
            E_t,
            elig,
            l2,
            H_t0,
            H_t0_t,
            H_scratch,
            A,
            B, 
            W_pred,
            W_act,
            a_t,
            U_t1,
            B_t,
            X_g0,
            X_g1,
            scratch0,
            scratch1,
            M,
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
            gw_DA,
            w2_NE,
            n,
            k,
            r,
            Ds,
            ad
        )
        stream.advance()
    }

    func learn(){
        get_eta(
            stream: stream,
            eta,
            elig,
            eligmean,
            scratch0,
            scratch1,
            n,
            k,
            eta_max,
            alpha_eta,
            DA_c
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
            elig,
            H_t1,
            H_t1_t,
            r,
            n,
            k,
            lambda,
            DA_c
        )
        stream.advance()
    }
    func update_chemicals(
        prediction_error: GPUBuffer<Float>
    ) {
        update_chemicals_public(
            prediction_error,
            scratch2,
            scratch3,
            err_sign,
            err_abs,
            err_sq,
            Ds,
            &DA_c,
            &ACh_c,
            &NE_c,
            d_DA,
            d_ACh,
            d_NE,
            k_DA,
            k_ACh,
            k_NE
        )
    }
    func zero_state(){
        zero(stream: stream, self.H_t0, n*k, 1)
        zero(stream: stream, self.H_t1, n*k, 1)
        zero(stream: stream, mu, n, 1)
        zero(stream: stream, gamma, n, 1)
        zero(stream: stream, U_t1, Ds, 1)
        zero(stream: stream, a_t, ad, 1)
        zero(stream: stream, elig, n*k, 1)
        DA_c=0
        ACh_c=0
        NE_c=0
        stream.advance()
    }
}
public func cortex_setup(
    n: UInt32,
    k: UInt32,
    r: UInt32,
    Ds: UInt32,
    ad: UInt32,
    softlog_alpha: Float,
    mes_alpha: Float,
    inhib_alpha: Float,
    lambda: Float,
    l2: Float,
    eta_max: Float,
    oja_bias: Float,
    w2_NE: Float,
    dt: Float,
    alpha_gamma: Float,
    leak: Float,
    gw_DA: Float,
    alpha_eta: Float,
    d_DA: Float,
    d_ACh:  Float,
    d_NE: Float,
    k_DA: Float,
    k_ACh: Float,
    k_NE: Float,
    eta_pred: Float
) -> CortexPrime{
    let B = GPUBuffer<Float>(device: device, capacity: Int(n*r))

    let W_pred = GPUBuffer<Float>(device: device, capacity: Int(Ds)*Int(n)*Int(k))
    let W_act = GPUBuffer<Float>(device: device, capacity: Int(ad)*Int(n)*Int(k))
    let W_conv_r3 = GPUBuffer<Float>(device: device, capacity: 7)
    let W_conv_r5 = GPUBuffer<Float>(device: device, capacity: 11)
    let W_conv_r7 = GPUBuffer<Float>(device: device, capacity: 15)
    let W_conv_r11 = GPUBuffer<Float>(device: device, capacity: 23)

    let W_inhib_sub_r3 = GPUBuffer<Float>(device: device, capacity: 7)
    let W_inhib_div_r7 = GPUBuffer<Float>(device: device, capacity: 15)

    let beta = GPUBuffer<Float>(device: device, capacity: Int(n))
    fill(stream: stream, beta, Float(1.5), n, 1)
    fill_random(stream: stream, W_pred, Ds*n*k, 1, -0.01 / sqrt(Float(n * k)), 0.01 / sqrt(Float(n * k)))
    fill_random(stream: stream, W_act, ad*n*k, 1, -0.01 / sqrt(Float(n * k)), 0.01 / sqrt(Float(n * k)))
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
    fill_random(stream: stream, B, n*r, 1, -0.005 / sqrt(Float(n)), 0.005 / sqrt(Float(n)))
    let cortex = CortexPrime(
        device: device,
        stream: stream,
        n: n,
        k: k,
        r: r,
        Ds: Ds,
        ad: ad,
        B: B,
        W_pred: W_pred,
        W_act: W_act, 
        W_conv_r3: W_conv_r3,
        W_conv_r5: W_conv_r5,
        W_conv_r7: W_conv_r7,
        W_conv_r11: W_conv_r11,
        W_inhib_sub_r3: W_inhib_sub_r3,
        W_inhib_div_r7: W_inhib_div_r7,
        beta: beta,
        softlog_alpha: softlog_alpha,
        mes_alpha: mes_alpha,
        inhib_alpha: inhib_alpha,
        lambda: lambda,
        l2: l2,
        eta_max: eta_max,
        oja_bias: oja_bias,
        w2_NE: w2_NE,
        dt: dt,
        alpha_gamma: alpha_gamma,
        leak: leak,
        gw_DA: gw_DA,
        alpha_eta: alpha_eta,
        d_DA: d_DA,
        d_ACh: d_ACh,
        d_NE: d_NE,
        k_DA: k_DA,
        k_ACh: k_ACh,
        k_NE: k_NE,
        eta_pred: eta_pred
    )
    /*
    softlog_alpha: 1.2,
    mes_alpha: 0.12,
    inhib_alpha: 0.9,
    lambda: 2e-5,
    l2: 0.97,
    eta_max: 1e-4,
    oja_bias: 0.0,
    w2_NE: 0.03,
    dt: 0.02,
    alpha_gamma: 0.15,
    leak: 0.005,
    gw_DA: 1.0,
    alpha_eta: 0.1
    */
    fill_random(stream: stream, cortex.A, n*r, 1, -0.01 / sqrt(Float(r)), 0.01 / sqrt(Float(r)))
    fill_random(stream: stream, cortex.H_t0, n*k, 1, -0.01, 0.01)
    return cortex
}