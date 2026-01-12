import Metal
import Foundation
import Darwin
let n: UInt32 = 2048
let k: UInt32 = 128
let r: UInt32 = 256
let m: UInt32 = 64
let Ds: UInt32 = 128
let ad: UInt32 = 16
let hr: UInt32 = 256
let context=MetalContext(kernelsDirectory: URL(fileURLWithPath: "../kernels"))
let device=context.device
let stream = ComputeStream(context: context)
public final class Cerebrum{
    var DA_c: Float=0
    var NE_c: Float=0
    var ACh_c: Float=0
    let d_NE: Float
    let d_ACh: Float
    let d_DA: Float
    let k_NE: Float
    let k_ACh: Float
    let k_DA: Float
    var prediction_error: GPUBuffer<Float>
    var scratch0: GPUBuffer<Float>
    var scratch1: GPUBuffer<Float>
    var err_sign: GPUBuffer<Float>
    var err_abs: GPUBuffer<Float>
    var err_sq: GPUBuffer<Float>
    let Cortex: CortexPrime
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
        self.prediction_error=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.scratch0=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.scratch1=GPUBuffer<Float>(device: device, capacity: Int(n*k))
        self.err_sign=GPUBuffer<Float>(device: device, capacity: 1)
        self.err_abs=GPUBuffer<Float>(device: device, capacity: 1)
        self.err_sq=GPUBuffer<Float>(device: device, capacity: 1)
        self.Cortex=cortex_setup()
        self.Thalamus=thalamus_setup()
        self.BasalGanglia=bg_setup()
        self.Hippocampus=hippocampus_setup()
        self.Cortex.zero_state()
        self.Thalamus.zero_state()
        self.Hippocampus.zero_state()
    }
    func update_chemicals(){
        mean_simd(stream: stream, prediction_error, scratch0, scratch1, err_sign, n*k, 1)
        abs_mean_simd(stream: stream, prediction_error, scratch0, scratch1, err_abs, n*k, 1)
        sqmean_simd(stream: stream, prediction_error, scratch0, scratch1, err_sq, n*k, 1)
        stream.advance()
        stream.synchronize()
        DA_c=d_DA*DA_c+(1-d_DA)*tanh(err_sign.ptr()[0]*k_DA)
        NE_c=d_NE*NE_c+(1-d_NE)*tanh(err_abs.ptr()[0]*k_NE)
        ACh_c=d_ACh*ACh_c+(1-d_ACh)*tanh(err_sq.ptr()[0]*k_ACh)
    }
    func step(
        U_t: GPUBuffer<Float>
    ){
        sub(stream: stream, Cortex.U_t1, U_t, prediction_error, n*k, 1)
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
