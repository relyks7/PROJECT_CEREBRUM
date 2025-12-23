import Metal
import Foundation
import Darwin
public func cortex_setup() -> (CortexPrime, GPUBuffer<Float>, ComputeStream, UInt32, UInt32){
    let cortex_ctx = MetalContext(
        kernelsDirectory: URL(fileURLWithPath: "../kernels")
    )
    let cortex_stream = ComputeStream(context: cortex_ctx)
    let device = cortex_ctx.device

    let n: UInt32 = 128
    let k: UInt32 = 16
    let r: UInt32 = 32

    let B = GPUBuffer<Float>(device: device, capacity: Int(n*r))

    let W_conv_r3  = GPUBuffer<Float>(device: device, capacity: 7)
    let W_conv_r5  = GPUBuffer<Float>(device: device, capacity: 11)
    let W_conv_r7  = GPUBuffer<Float>(device: device, capacity: 15)
    let W_conv_r11 = GPUBuffer<Float>(device: device, capacity: 23)

    let W_inhib_sub_r3 = GPUBuffer<Float>(device: device, capacity: 7)
    let W_inhib_div_r7 = GPUBuffer<Float>(device: device, capacity: 15)

    let beta   = GPUBuffer<Float>(device: device, capacity: Int(n))
    let test_E = GPUBuffer<Float>(device: device, capacity: Int(n*k))

    zero(stream: cortex_stream, B, n*r, 1)
    zero(stream: cortex_stream, beta, n, 1)
    zero(stream: cortex_stream, test_E, n*k, 1)
    zero(stream: cortex_stream, W_conv_r3, 7, 1)
    zero(stream: cortex_stream, W_conv_r5, 11, 1)
    zero(stream: cortex_stream, W_conv_r7, 15, 1)
    zero(stream: cortex_stream, W_conv_r11, 23, 1)
    zero(stream: cortex_stream, W_inhib_sub_r3, 7, 1)
    zero(stream: cortex_stream, W_inhib_div_r7, 15, 1)
    for i in 0..<Int(n) {
        beta.ptr()[i] = 0.5
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
        B.ptr()[Int(i)] = Float.random(in: -0.05...0.05)
    }

    cortex_stream.synchronize()

    let cortex = CortexPrime(
        device: device,
        stream: cortex_stream,
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
        softlog_alpha: 1.7,
        mes_alpha: 0.05,
        inhib_alpha: 1.0,
        lambda: 1e-4,
        eta_max: 1e-2,
        oja_bias: 0.0,
        w_NE: 0.5,
        w_ACh: 1.0,
        w_DA: 2.0
    )
    for i in 0..<(n*r) {
        cortex.A.ptr()[Int(i)] = Float.random(in: -0.02...0.02)
    }
    return (cortex, test_E, cortex_stream, n, k)
}
let (cortex, test_E, stream, n, k) = cortex_setup()

// --------------------------------------------------
// ATTRACTOR TEST (OUTSIDE SETUP)
// --------------------------------------------------

// Allocate buffer to store previous state
let H_prev = GPUBuffer<Float>(
    device: cortex.device,
    capacity: Int(n * k)
)

// 1. Inject a brief random pulse
for i in 0..<Int(n * k) {
    test_E.ptr()[i] = Float.random(in: -0.5...0.5)
}

// 2. Apply pulse once
cortex.step(E_t: test_E)
stream.synchronize()

// 3. Remove input
zero(stream: stream, test_E, n * k, 1)
stream.synchronize()

// 4. Run free dynamics
let steps = 100

for t in 0..<steps {

    // --- SNAPSHOT PREVIOUS STATE ---
    copy(stream: stream, cortex.H_t0, H_prev, n * k, 1)
    stream.synchronize()

    // --- ADVANCE DYNAMICS ---
    cortex.step(E_t: test_E)
    stream.synchronize()

    // --- MEASURE CHANGE ---
    let h0 = H_prev.ptr()
    let h1 = cortex.H_t0.ptr()

    var delta: Float = 0.0
    for i in 0..<Int(n * k) {
        let d = h1[i] - h0[i]
        delta += d * d
    }

    print("step \(t): ΔH =", delta)
}