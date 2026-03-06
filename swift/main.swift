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
    let CortexL1: CortexPrime
    let CortexL2: CortexPrime
    let CortexL3: CortexPrime
    let Thalamus: ThalamusPrime
    let BasalGanglia: BasalGangliaPrime
    let Hippocampus: HippocampusPrime
    init(){
        self.d_NE=0.995
        self.d_ACh=0.998
        self.d_DA=0.997
        self.k_NE=2.0
        self.k_ACh=5.0
        self.k_DA=20.0
        self.CortexL1=cortex_setup(
                        n: 2048, k: 128, r: 256, Ds: 128, ad: 16,
                        softlog_alpha: 0.8,
                        mes_alpha: 0.08,
                        inhib_alpha: 1.05,
                        lambda: 5e-5,
                        l2: 0.93,
                        eta_max: 2e-4,
                        oja_bias: 0.0,
                        w2_NE: 0.06,
                        dt: 0.03,
                        alpha_gamma: 0.12,
                        leak: 0.010,
                        gw_DA: 0.7,
                        alpha_eta: 0.16,
                        d_DA: 0.992,
                        d_ACh: 0.995,
                        d_NE: 0.990,
                        k_DA: 16,
                        k_ACh: 4,
                        k_NE: 1.5
                    )

        self.CortexL2=cortex_setup(
                        n: 1536, k: 96, r: 192, Ds: 128, ad: 16,
                        softlog_alpha: 1.2,
                        mes_alpha: 0.14,
                        inhib_alpha: 0.90,
                        lambda: 2e-5,
                        l2: 0.97,
                        eta_max: 1e-4,
                        oja_bias: 0.0,
                        w2_NE: 0.03,
                        dt: 0.020,
                        alpha_gamma: 0.15,
                        leak: 0.005,
                        gw_DA: 1.0,
                        alpha_eta: 0.10,
                        d_DA: 0.997,
                        d_ACh: 0.998,
                        d_NE: 0.995,
                        k_DA: 20,
                        k_ACh: 5,
                        k_NE: 2
                    )
        self.CortexL3=cortex_setup(
                        n: 1024, k: 64, r: 128, Ds: 128, ad: 16,
                        softlog_alpha: 1.5,
                        mes_alpha: 0.10,
                        inhib_alpha: 1.10,
                        lambda: 1e-5,
                        l2: 0.985,
                        eta_max: 5e-5,
                        oja_bias: 0.0,
                        w2_NE: 0.015,
                        dt: 0.012,
                        alpha_gamma: 0.20,
                        leak: 0.003,
                        gw_DA: 1.3,
                        alpha_eta: 0.07,
                        d_DA: 0.999,
                        d_ACh: 0.999,
                        d_NE: 0.998,
                        k_DA: 18,
                        k_ACh: 4,
                        k_NE: 1.5
                    )
        self.Thalamus=thalamus_setup()
        self.BasalGanglia=bg_setup()
        self.Hippocampus=hippocampus_setup()
        self.Cortex.zero_state()
        self.Thalamus.zero_state()
        self.Hippocampus.zero_state()
        self.prediction_error=GPUBuffer<Float>(device: device, capacity: Int(Ds))
        self.scratch0=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.scratch1=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.err_sign=GPUBuffer<Float>(device: device, capacity: 1)
        self.err_abs=GPUBuffer<Float>(device: device, capacity: 1)
        self.err_sq=GPUBuffer<Float>(device: device, capacity: 1)
    }
    func step(
        U_t_1: GPUBuffer<Float>
    ){
        sub(stream: stream, Cortex.U_t1, U_t, prediction_error, Ds, 1)
        update_chemicals()
        BasalGanglia.step(
            H_t0: Cortex.H_t0,
            a_t: Cortex.a_t,
            DA_c: DA_c
        )
        Thalamus.step(
            H_t0: Cortex.H_t0,
            g: BasalGanglia.g,
            U_t: U_t
        )
        Cortex.step(
            E_t: Thalamus.E_t,
            NE_c: NE_c,
            ACh_c: ACh_c,
            DA_c: DA_c
        )
        Hippocampus.step(
            H_t: Cortex.H_t0,
            DA_c: DA_c,
            ACh_c: ACh_c,
            NE_c: NE_c
        )
    }
    func cortex_learn(){
        Cortex.learn(
            DA_c: DA_c
        )
    }
    func bg_learn(){
        BasalGanglia.learn(
            H_t0: Cortex.H_t0,
            DA_c: DA_c
        )
    }
    func recall_step(
        steps: UInt32
    ){
        for _ in 0..<steps {
            Hippocampus.step(
                H_t: Cortex.H_t0,
                DA_c: DA_c,
                ACh_c: 1.0,
                NE_c: 0.1
            )
            Cortex.learn(
                DA_c: DA_c
            )
        }
    }
}
//Sextus est discipulus malus!
let Sextus=Cerebrum()
