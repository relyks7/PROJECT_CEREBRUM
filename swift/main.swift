import Metal
import Foundation
import Darwin
let n: UInt32 = 2048
let k: UInt32 = 128
let r: UInt32 = 256
let m: UInt32 = 64
let Ds: UInt32 = 128
let context=MetalContext(kernelsDirectory: URL(fileURLWithPath: "../kernels"))
let device=context.device
let stream = ComputeStream(context: context)
public final class Cerebrum{
    var NE: GPUBuffer<Float>
    var ACh: GPUBuffer<Float>
    var DA: GPUBuffer<Float>
    let d_NE: Float
    let d_ACh: Float
    let d_DA: Float
    let Cortex: CortexPrime
    let Thalamus: ThalamusPrime
    init(){
        NE=GPUBuffer(device: device, capacity: Int(n))
        ACh=GPUBuffer(device: device, capacity: Int(n))
        d_NE=0.995
        d_ACh=0.998
        d_DA=0.997
        DA=GPUBuffer(device: device, capacity: Int(n))
        Cortex=cortex_setup()
        Thalamus=thalamus_setup()
        Cortex.zero_state()
        Thalamus.zero_state()
    }
    func chem_decay() {
        for i in 0..<Int(n) {
            NE.ptr()[i]  *= d_NE
            ACh.ptr()[i] *= d_ACh
            DA.ptr()[i]  *= d_DA
        }
    }
    func step(
        U_t: GPUBuffer<Float>
    ){
        chem_decay()
        Thalamus.chem_mean(
            NE,
            ACh,
            DA
        )
        Thalamus.step(
            Cortex.H_t0,
            U_t
        )
        Cortex.step(Thalamus.E_t)
    }
    func learn(){
        Cortex.learn(
            NE,
            ACh,
            DA
        )
    }
}