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
    zero(stream: cortex_stream, beta, n, 1)
    zero(stream: cortex_stream, test_E, n*k, 1)
    zero(stream: cortex_stream, W_conv_r3, 7, 1)
    zero(stream: cortex_stream, W_conv_r5, 11, 1)
    zero(stream: cortex_stream, W_conv_r7, 15, 1)
    zero(stream: cortex_stream, W_conv_r11, 23, 1)
    zero(stream: cortex_stream, W_inhib_sub_r3, 7, 1)
    zero(stream: cortex_stream, W_inhib_div_r7, 15, 1)
    cortex_stream.advance()
    cortex_stream.synchronize()
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
        softlog_alpha: 0.8,
        mes_alpha: 0.08,
        inhib_alpha: 1.2,
        lambda: 1e-4,
        eta_max: 1e-2,
        oja_bias: 0.0,
        w_NE: 0.5,
        w_ACh: 1.0,
        w_DA: 2.0,
        dt: 0.02,
        alpha_gamma: 1.0,
        leak: 0.003
    )
    cortex_stream.advance()
    cortex_stream.synchronize()
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
    return (cortex, test_E, cortex_stream, n, k)
}
let (cortex, test_E, stream, n, k) = cortex_setup()
// ==================================================
// CORTEX VALIDATION SUITE
// ==================================================

@inline(__always)
func l2(_ a: GPUBuffer<Float>, _ b: GPUBuffer<Float>, count: Int) -> Float {
    var s: Float = 0
    let pa = a.ptr()
    let pb = b.ptr()
    for i in 0..<count {
        let d = pa[i] - pb[i]
        s += d * d
    }
    return sqrt(s)
}

@inline(__always)
func cosine(_ a: GPUBuffer<Float>, _ b: GPUBuffer<Float>, count: Int) -> Float {
    var dot: Float = 0
    var na: Float = 0
    var nb: Float = 0
    let pa = a.ptr()
    let pb = b.ptr()
    for i in 0..<count {
        let x = pa[i]
        let y = pb[i]
        dot += x * y
        na += x * x
        nb += y * y
    }
    return dot / (sqrt(na * nb) + 1e-8)
}

let countI = Int(n * k)
let countU = UInt32(n * k)

// --------------------------------------------------
// TEST 1: FIXED POINT CONVERGENCE
// --------------------------------------------------
var prevΔ: Float = .infinity
var converges = true

for _ in 0..<100 {
    cortex.step(E_t: test_E)
    stream.advance()
    stream.synchronize()

    let d = l2(cortex.H_t0, cortex.H_t1, count: countI)
    if d > prevΔ { converges = false }
    prevΔ = d
}
print("Convergence:", converges)

// --------------------------------------------------
// TEST 2: PERTURBATION RECOVERY
// --------------------------------------------------
let H_star = GPUBuffer<Float>(device: cortex.device, capacity: countI)
copy(stream: stream, cortex.H_t0, H_star, countU, 1)
stream.advance()
stream.synchronize()

for i in 0..<countI {
    cortex.H_t0.ptr()[i] += Float.random(in: -0.1...0.1)
}

for _ in 0..<150 {
    cortex.step(E_t: test_E)
}
stream.advance()
stream.synchronize()

let recovery = l2(cortex.H_t0, H_star, count: countI) < 0.05
print("Perturbation Recovery:", recovery)

// --------------------------------------------------
// TEST 3: INPUT EQUIVALENCE (SHIFT INVARIANCE)
// --------------------------------------------------
let E_shift = GPUBuffer<Float>(device: cortex.device, capacity: countI)
let pk = Int(k)
let pn = Int(n)

for y in 0..<pn {
    for x in 0..<pk {
        E_shift.ptr()[y * pk + x] =
            test_E.ptr()[((y + 3) % pn) * pk + x]
    }
}

cortex.zero_state()
for _ in 0..<150 {
    cortex.step(E_t: test_E)
}
stream.advance()
stream.synchronize()

let H1 = GPUBuffer<Float>(device: cortex.device, capacity: countI)
copy(stream: stream, cortex.H_t0, H1, countU, 1)

cortex.zero_state()
for _ in 0..<150 {
    cortex.step(E_t: E_shift)
}
stream.advance()
stream.synchronize()

let equiv = cosine(H1, cortex.H_t0, count: countI) > 0.7
print("Input Equivalence:", equiv)

// --------------------------------------------------
// TEST 4: HYSTERESIS (WORKING MEMORY)
// --------------------------------------------------
let E_alt = GPUBuffer<Float>(device: cortex.device, capacity: countI)
for i in 0..<countI {
    E_alt.ptr()[i] = Float.random(in: -0.5...0.5)
}

cortex.zero_state()
for _ in 0..<120 { cortex.step(E_t: test_E) }
for _ in 0..<20  { cortex.step(E_t: E_alt) }
for _ in 0..<120 { cortex.step(E_t: test_E) }

stream.advance()
stream.synchronize()

let hysteresis = l2(cortex.H_t0, H_star, count: countI) > 1e-3
print("Hysteresis:", hysteresis)

// --------------------------------------------------
// TEST 5: ENERGY DESCENT
// --------------------------------------------------
var energyOK = true
var prevE: Float = .infinity

for _ in 0..<80 {
    cortex.step(E_t: test_E)
    stream.advance()
    stream.synchronize()

    let e = l2(cortex.H_t0, cortex.H_t1, count: countI)
    if e > prevE { energyOK = false }
    prevE = e
}
print("Energy Descent:", energyOK)

// ==================================================
// FINAL VERDICT
// ==================================================
let cortexWorks = converges && recovery && equiv && hysteresis && energyOK
print("CORTEX FUNCTIONAL:", cortexWorks)