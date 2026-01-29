import Metal
import Foundation
import Darwin
public func chemical_compress(
    stream: ComputeStream,
    _ NE_: GPUBuffer<Float>,
    _ ACh_: GPUBuffer<Float>,
    _ DA_: GPUBuffer<Float>,
    _ mean_scratch_0: GPUBuffer<Float>,
    _ mean_scratch_1: GPUBuffer<Float>,
    _ NE_0: GPUBuffer<Float>,
    _ ACh_0: GPUBuffer<Float>,
    _ DA_0: GPUBuffer<Float>,
    _ NE_c: inout Float,
    _ ACh_c: inout Float,
    _ DA_c: inout Float,
    _ n_: UInt32
){
    mean_simd(stream: stream, NE_, mean_scratch_0, mean_scratch_1, NE_0, n_, 1)
    mean_simd(stream: stream, ACh_, mean_scratch_0, mean_scratch_1, ACh_0, n_, 1)
    mean_simd(stream: stream, DA_, mean_scratch_0, mean_scratch_1, DA_0, n_, 1)
    stream.advance()
    stream.synchronize()
    NE_c  = max(0.0, min(NE_0.ptr()[0],  2.0))
    ACh_c = max(0.0, min(ACh_0.ptr()[0], 2.0))
    DA_c  = max(0.0, min(DA_0.ptr()[0],  2.0))
}
public func thalamus_step(
    stream: ComputeStream,
    _ H_t0: GPUBuffer<Float>,
    _ z_t0: GPUBuffer<Float>,
    _ z_t1: GPUBuffer<Float>,
    _ z_scratch0: GPUBuffer<Float>,
    _ z_scratch1: GPUBuffer<Float>,
    _ z_scratch2: GPUBuffer<Float>,
    _ g: GPUBuffer<Float>,
    _ E_t: GPUBuffer<Float>,
    _ W_tc: GPUBuffer<Float>,
    _ W_cx: GPUBuffer<Float>,
    _ U_t: GPUBuffer<Float>,
    _ W_s: GPUBuffer<Float>,
    _ m_: UInt32,
    _ n_: UInt32,
    _ k_: UInt32,
    _ Ds_: UInt32,
    _ lambda_t: Float,
    _ w_ACh_0: Float,
    _ w_ACh_1: Float,
    _ w_NE_: Float,
    _ NE_c: Float,
    _ ACh_c: Float,
    _ DA_c: Float
){
    gemv(stream: stream, W_s, U_t, z_scratch0, m_, Ds_)
    gemv(stream: stream, W_cx, H_t0, z_scratch1, m_, n_*k_)
    add_scaled(stream: stream, z_scratch0, z_scratch1, z_scratch2, m_, 1, (1+w_NE_*NE_c)*(1+w_ACh_0*ACh_c), (1+w_NE_*NE_c)*(1-w_ACh_1*ACh_c))
    dsb(stream: stream, z_t0, z_scratch2, z_t1, g, m_, 1, lambda_t*exp(-DA_c))
    gemv(stream: stream, W_tc, z_t0, E_t, n_*k_, m_)
    copy(stream: stream, z_t1, z_t0, m_, 1)
}
public final class ThalamusPrime{
    //gpu
    let device: MTLDevice
    let stream: ComputeStream
    //main
    var z_t0: GPUBuffer<Float>
    var z_t1: GPUBuffer<Float>
    var E_t: GPUBuffer<Float>
    var W_tc: GPUBuffer<Float>
    var W_cx: GPUBuffer<Float>
    var W_s: GPUBuffer<Float>
    var NE_c: Float
    var ACh_c: Float
    var DA_c: Float
    //scratch
    var z_scratch0: GPUBuffer<Float>
    var z_scratch1: GPUBuffer<Float>
    var z_scratch2: GPUBuffer<Float>
    var mean_scratch_0: GPUBuffer<Float>
    var mean_scratch_1: GPUBuffer<Float>
    var NE_0: GPUBuffer<Float>
    var ACh_0: GPUBuffer<Float>
    var DA_0: GPUBuffer<Float>
    //params
    let m: UInt32
    let n: UInt32
    let k: UInt32
    let Ds: UInt32
    let lambda_t: Float
    let w_ACh_0: Float
    let w_ACh_1: Float
    let w_NE: Float
    init(
        device: MTLDevice, 
        stream: ComputeStream,
        W_tc: GPUBuffer<Float>,
        W_cx: GPUBuffer<Float>,
        W_s: GPUBuffer<Float>,
        m: UInt32,
        n: UInt32,
        k: UInt32,
        Ds: UInt32,
        lambda_t: Float,
        w_ACh_0: Float,
        w_ACh_1: Float,
        w_NE: Float){
        self.z_t0=GPUBuffer(device: device, capacity: Int(m))
        self.z_t1=GPUBuffer(device: device, capacity: Int(m))
        self.E_t=GPUBuffer(device: device, capacity: Int(n*k))
        self.z_scratch0=GPUBuffer(device: device, capacity: Int(m))
        self.z_scratch1=GPUBuffer(device: device, capacity: Int(m))
        self.z_scratch2=GPUBuffer(device: device, capacity: Int(m))
        self.mean_scratch_0=GPUBuffer(device: device, capacity: Int(n))
        self.mean_scratch_1=GPUBuffer(device: device, capacity: Int(n))
        self.NE_0=GPUBuffer(device: device, capacity: Int(1))
        self.ACh_0=GPUBuffer(device: device, capacity: Int(1))
        self.DA_0=GPUBuffer(device: device, capacity: Int(1))
        self.device=device
        self.stream=stream
        self.W_tc=W_tc
        self.W_cx=W_cx
        self.W_s=W_s
        self.m=m
        self.n=n
        self.k=k
        self.Ds=Ds
        self.lambda_t=lambda_t
        self.w_ACh_0=w_ACh_0
        self.w_ACh_1=w_ACh_1
        self.w_NE=w_NE
        self.NE_c  = 0.0
        self.ACh_c = 0.0
        self.DA_c  = 0.0
    }
    func step(
        H_t0: GPUBuffer<Float>,
        g: GPUBuffer<Float>,
        U_t: GPUBuffer<Float>
    )
    {
        thalamus_step(
            stream: stream,
            H_t0,
            z_t0,
            z_t1,
            z_scratch0,
            z_scratch1,
            z_scratch2,
            g,
            E_t,
            W_tc,
            W_cx,
            U_t,
            W_s,
            m,
            n,
            k,
            Ds,
            lambda_t,
            w_ACh_0,
            w_ACh_1,
            w_NE,
            NE_c,
            ACh_c,
            DA_c
        )
        stream.advance()
    }
    func zero_state(){
        zero(stream: stream, self.z_t0, m, 1)
        zero(stream: stream, self.z_t1, m, 1)
        zero(stream: stream, self.E_t, n*k, 1)
        stream.advance()
    }
}
public func thalamus_setup(
    m: UInt32,
    n: UInt32,
    k: UInt32,
    lambda_t: Float,
    w_ACh_0: Float,
    w_ACh_1: Float,
    w_NE: Float
) -> ThalamusPrime{
    let W_cx = GPUBuffer<Float>(
        device: device,
        capacity: Int(m) * Int(n * k)
    )
    let W_tc = GPUBuffer<Float>(
        device: device,
        capacity: Int(n * k) * Int(m)
    )
    let W_s = GPUBuffer<Float>(
        device: device,
        capacity: Int(m) * Int(Ds)
    )
    let scale_cx = 1.0 / sqrt(Float(n * k))
    let scale_tc = 1.0 / sqrt(Float(m))
    let scale_s  = 1.0 / sqrt(Float(Ds))
    for i in 0..<W_cx.count {
        W_cx.ptr()[i] = Float.random(in: -1.0...1.0) * scale_cx
    }
    for i in 0..<W_tc.count {
        W_tc.ptr()[i] = Float.random(in: -1.0...1.0) * scale_tc
    }
    for i in 0..<W_s.count {
        W_s.ptr()[i] = Float.random(in: -1.0...1.0) * scale_s
    }
    let thalamus=ThalamusPrime(
        device: device,
        stream: stream,
        W_tc: W_tc,
        W_cx: W_cx,
        W_s: W_s,
        m: m,
        n: n,
        k: k,
        Ds: Ds,
        lambda_t: lambda_t,
        w_ACh_0: w_ACh_0,
        w_ACh_1: w_ACh_1,
        w_NE: w_NE
    )
    /*
    lambda_t: 0.02,
    w_ACh_0: 0.6,
    w_ACh_1: 0.4,
    w_NE: 0.5
    */
    return thalamus
}