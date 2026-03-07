import Metal
import Foundation
import Darwin
/*
let n: UInt32 = 2048
let k: UInt32 = 128
let r: UInt32 = 256
let m: UInt32 = 64
let Ds: UInt32 = 128
let ad: UInt32 = 16
let hr: UInt32 = 256
*/
let context=MetalContext(kernelsDirectory: URL(fileURLWithPath: "../kernels"))
let device=context.device
let stream = ComputeStream(context: context)
public final class Cerebrum{
    let U_t: GPUBuffer<Float>
    let CortexL1: CortexPrime
    let CortexL2: CortexPrime
    let CortexL3: CortexPrime
    let L1_bottlenecked: GPUBuffer<Float>
    let L2_bottlenecked: GPUBuffer<Float>
    let ThalamusL1: ThalamusPrime
    let ThalamusL2: ThalamusPrime
    let ThalamusL3: ThalamusPrime
    let BasalGangliaL1: BasalGangliaPrime
    let BasalGangliaL2: BasalGangliaPrime
    let BasalGangliaL3: BasalGangliaPrime
    let Hippocampus: HippocampusPrime
    let pred_error_l1: GPUBuffer<Float>
    let pred_error_l2: GPUBuffer<Float>
    let pred_error_l3: GPUBuffer<Float>
    let p1: UInt32
    let p2: UInt32
    let W_bottleneck_L1: GPUBuffer<Float>
    let W_bottleneck_L2: GPUBuffer<Float>
    init(){
        self.p1=128
        self.p2=64
        self.CortexL1 = cortex_setup(
            n: 512, k: 32, r: 64, Ds: 2, ad: 16,
            softlog_alpha: 1.0,
            mes_alpha: 0.01,
            inhib_alpha: 1.05,
            lambda: 5e-4,
            l2: 0.92,
            eta_max: 1e-6,
            oja_bias: 0.0,
            w2_NE: 0.03,
            dt: 0.06,
            alpha_gamma: 0.12,
            leak: 0.15,
            gw_DA: 0.3,
            alpha_eta: 0.08,
            d_DA: 0.970,
            d_ACh: 0.950,
            d_NE: 0.970,
            k_DA: 1.1,
            k_ACh: 1.5,
            k_NE: 0.3,
            eta_pred: 0.1
        )
        self.U_t=GPUBuffer<Float>(device: device, capacity: Int(CortexL1.Ds))
        self.CortexL2 = cortex_setup(
            n: 256, k: 16, r: 32, Ds: p1, ad: 16,
            softlog_alpha: 1.0,
            mes_alpha: 0.06,
            inhib_alpha: 0.90,
            lambda: 2e-4,
            l2: 0.95,
            eta_max: 5e-5,
            oja_bias: 0.0,
            w2_NE: 0.02,
            dt: 0.02,
            alpha_gamma: 0.12,
            leak: 0.03,
            gw_DA: 0.4,
            alpha_eta: 0.06,
            d_DA: 0.985,
            d_ACh: 0.980,
            d_NE: 0.985,
            k_DA: 0.8,
            k_ACh: 2.0,
            k_NE: 0.3,
            eta_pred: 0.01
        )
        self.CortexL3 = cortex_setup(
            n: 128, k: 8, r: 16, Ds: p2, ad: 16,
            softlog_alpha: 1.0,
            mes_alpha: 0.08,
            inhib_alpha: 0.90,
            lambda: 1e-4,
            l2: 0.97,
            eta_max: 1e-5,
            oja_bias: 0.0,
            w2_NE: 0.01,
            dt: 0.005,
            alpha_gamma: 0.12,
            leak: 0.005,
            gw_DA: 0.4,
            alpha_eta: 0.04,
            d_DA: 0.990,
            d_ACh: 0.990,
            d_NE: 0.990,
            k_DA: 6.0,
            k_ACh: 2.0,
            k_NE: 0.3,
            eta_pred: 0.0005
        )
        self.L1_bottlenecked=GPUBuffer<Float>(device: device, capacity: Int(p1))
        self.L2_bottlenecked=GPUBuffer<Float>(device: device, capacity: Int(p2))
        self.W_bottleneck_L1=GPUBuffer<Float>(device: device, capacity: Int(p1*CortexL1.n*CortexL1.k))
        self.W_bottleneck_L2=GPUBuffer<Float>(device: device, capacity: Int(p2*CortexL2.n*CortexL2.k))
        fill_random(stream: stream, W_bottleneck_L1, p1*CortexL1.n*CortexL1.k, 1, -0.01, 0.01)
        fill_random(stream: stream, W_bottleneck_L2, p2*CortexL2.n*CortexL2.k, 1, -0.01, 0.01)
        self.ThalamusL1 = thalamus_setup(
            m: 32,
            n: CortexL1.n,
            k: CortexL1.k,
            Ds: CortexL1.Ds,
            lambda_t: 0.08,
            w_ACh_0: 0.8,
            w_ACh_1: 0.6,
            w_NE: 0.3
        )
        self.ThalamusL2 = thalamus_setup(
            m: 32,
            n: CortexL2.n,
            k: CortexL2.k,
            Ds: CortexL2.Ds,
            lambda_t: 0.05,
            w_ACh_0: 0.6,
            w_ACh_1: 0.4,
            w_NE: 0.15
        )
        self.ThalamusL3 = thalamus_setup(
            m: 32,
            n: CortexL3.n,
            k: CortexL3.k,
            Ds: CortexL3.Ds,
            lambda_t: 0.02,
            w_ACh_0: 0.4,
            w_ACh_1: 0.2,
            w_NE: 0.05
        )
        self.BasalGangliaL1 = bg_setup(
            m: 32,
            n: CortexL1.n,
            k: CortexL1.k,
            ad: 16,
            alpha: 1.0,
            alpha_f: 0.8,
            beta: 1.0,
            kappa: 3.0,
            eta0: 5e-5,
            eta_max: 1e-4,
            gamma: 1.5,
            k_low: 0.6,
            k_high: 1.0
        )
        self.BasalGangliaL2 = bg_setup(
            m: 32,
            n: CortexL2.n,
            k: CortexL2.k,
            ad: 16,
            alpha: 1.2,
            alpha_f: 0.8,
            beta: 1.2,
            kappa: 3.0,
            eta0: 5e-6,
            eta_max: 5e-5,
            gamma: 1.5,
            k_low: 0.6,
            k_high: 1.0
        )
        self.BasalGangliaL3 = bg_setup(
            m: 32,
            n: CortexL3.n,
            k: CortexL3.k,
            ad: 16,
            alpha: 1.5,
            alpha_f: 0.8,
            beta: 1.5,
            kappa: 3.0,
            eta0: 5e-7,
            eta_max: 1e-5,
            gamma: 1.5,
            k_low: 0.6,
            k_high: 1.0
        )
        self.Hippocampus=hippocampus_setup(
            n: CortexL3.n,
            k: CortexL3.k,
            hr: 256,
            epsilon:0.002,
            alpha: 0.15,
            theta: 0.2,
            lambda: 0.01,
            tau_b: 0.03,
            omega: 2.0 * Float.pi / 120.0
        )
        self.pred_error_l1=GPUBuffer<Float>(device: device, capacity: Int(CortexL1.Ds))
        self.pred_error_l2=GPUBuffer<Float>(device: device, capacity: Int(CortexL2.Ds))
        self.pred_error_l3=GPUBuffer<Float>(device: device, capacity: Int(CortexL3.Ds))
        self.CortexL1.zero_state()
        self.CortexL2.zero_state()
        self.CortexL3.zero_state()
        self.ThalamusL1.zero_state()
        self.ThalamusL2.zero_state()
        self.ThalamusL3.zero_state()
        self.Hippocampus.zero_state()
    }
    func step(){
        gemv(stream: stream, W_bottleneck_L1, CortexL1.H_t0, L1_bottlenecked, p1, CortexL1.n*CortexL1.k)
        gemv(stream: stream, W_bottleneck_L2, CortexL2.H_t0, L2_bottlenecked, p2, CortexL2.n*CortexL2.k)
        sub(stream: stream, CortexL1.U_t1, U_t, pred_error_l1, CortexL1.Ds, 1)
        sub(stream: stream, CortexL2.U_t1, L1_bottlenecked, pred_error_l2, CortexL2.Ds, 1)
        sub(stream: stream, CortexL3.U_t1, L2_bottlenecked, pred_error_l3, CortexL3.Ds, 1)
        CortexL1.learn_pred(prediction_error:pred_error_l1)
        CortexL2.learn_pred(prediction_error:pred_error_l2)
        CortexL3.learn_pred(prediction_error:pred_error_l3)
        CortexL1.update_chemicals(prediction_error:pred_error_l1)
        CortexL2.update_chemicals(prediction_error:pred_error_l2)
        CortexL3.update_chemicals(prediction_error:pred_error_l3)
        BasalGangliaL1.step(
            H_t0: CortexL1.H_t0,
            a_t: CortexL1.a_t,
            DA_c: CortexL1.DA_c
        )
        ThalamusL1.step(
            H_t0: CortexL1.H_t0,
            g: BasalGangliaL1.g,
            U_t: U_t
        )
        BasalGangliaL2.step(
            H_t0: CortexL2.H_t0,
            a_t: CortexL2.a_t,
            DA_c: CortexL2.DA_c
        )
        ThalamusL2.step(
            H_t0: CortexL2.H_t0,
            g: BasalGangliaL2.g,
            U_t: L1_bottlenecked
        )
        BasalGangliaL3.step(
            H_t0: CortexL3.H_t0,
            a_t: CortexL3.a_t,
            DA_c: CortexL3.DA_c
        )
        ThalamusL3.step(
            H_t0: CortexL3.H_t0,
            g: BasalGangliaL3.g,
            U_t: L2_bottlenecked
        )
        CortexL1.step(
            E_t: ThalamusL1.E_t
        )
        CortexL2.step(
            E_t: ThalamusL2.E_t
        )
        CortexL3.step(
            E_t: ThalamusL3.E_t
        )
        Hippocampus.step(
            H_t: CortexL3.H_t0,
            DA_c: CortexL3.DA_c,
            ACh_c: CortexL3.ACh_c,
            NE_c: CortexL3.NE_c
        )
    }
    func cortex_learn(){
        CortexL1.learn()
        CortexL2.learn()
        CortexL3.learn()
    }
    func bg_learn(){
        BasalGangliaL1.learn(
            H_t0: CortexL1.H_t0,
            DA_c: CortexL1.DA_c
        )
        BasalGangliaL2.learn(
            H_t0: CortexL2.H_t0,
            DA_c: CortexL2.DA_c
        )
        BasalGangliaL3.learn(
            H_t0: CortexL3.H_t0,
            DA_c: CortexL3.DA_c
        )
    }
    func recall_step(
        steps: UInt32
    ){
        for _ in 0..<steps {
            Hippocampus.step(
                H_t: CortexL3.H_t0,
                DA_c: CortexL3.DA_c,
                ACh_c: 1.0,
                NE_c: 0.1
            )
            CortexL3.learn()
        }
    }
}
//Sextus est discipulus malus!
let Sextus=Cerebrum()
let frames=2400
var points:[(Float, Float)]=[]
for i in 0..<frames{
    let t=2.0*3.141592653589793*Float(i)/Float(frames)
    points.append((sin(t),sin(2*t)))
}
let ticks=3600000
for i in 0..<ticks {

    let (x, y) = points[i % frames]

    write_point(stream: stream, Sextus.U_t, x, y)

    Sextus.step()
    if i%20==0{
    Sextus.cortex_learn()}
    if i%5==0{
    Sextus.bg_learn()}

    if i % 101 == 0 { //temporary disable, temporal poisoning?
        Sextus.recall_step(steps: 8)
    }
    if i%20==0{
    stream.advance()
    stream.synchronize()

    let err1 = Sextus.CortexL1.err_abs.ptr()[0]
    let err2 = Sextus.CortexL2.err_abs.ptr()[0]
    let err3 = Sextus.CortexL3.err_abs.ptr()[0]

    let out = String(format:
        "[%05d] Target: (% .2f, % .2f) | Pred: (% .2f, % .2f)  Err: [%.4f, %.4f, %.4f] DA: [%.2f, %.2f, %.2f]",
        i, x, y,
        Sextus.CortexL1.U_t1.ptr()[0],
        Sextus.CortexL1.U_t1.ptr()[1],
        err1, err2, err3,
        Sextus.CortexL1.DA_c,
        Sextus.CortexL2.DA_c,
        Sextus.CortexL3.DA_c
    )

    print(out)

    if err1.isNaN || err2.isNaN || err3.isNaN {
        print("NaN")
        exit(1)
    }}
}