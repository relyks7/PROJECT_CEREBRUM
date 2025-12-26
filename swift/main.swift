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
    let BasalGanglia: BasalGangliaPrime
    init(){
        NE=GPUBuffer(device: device, capacity: Int(n))
        ACh=GPUBuffer(device: device, capacity: Int(n))
        d_NE=0.995
        d_ACh=0.998
        d_DA=0.997
        DA=GPUBuffer(device: device, capacity: Int(n))
        Cortex=cortex_setup()
        Thalamus=thalamus_setup()
        BasalGanglia=bg_setup()
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
            NE: NE,
            ACh: ACh,
            DA: DA
        )
        BasalGanglia.step(
            H_t0: Cortex.H_t0,
            DA_c: Thalamus.DA_c
        )
        Thalamus.step(
            H_t0: Cortex.H_t0,
            g: BasalGanglia.g,
            U_t: U_t
        )
        Cortex.step(
            E_t: Thalamus.E_t,
            NE_c: Thalamus.NE_c,
            ACh_c: Thalamus.ACh_c,
            DA_c: Thalamus.DA_c
        )
    }
    func cortex_learn(){
        Cortex.learn(
            NE: NE,
            ACh: ACh,
            DA: DA
        )
    }
    func bg_learn(){
        BasalGanglia.learn(
            H_t0: Cortex.H_t0,
            DA_c: Thalamus.DA_c
        )
    }
}
//Sextus est discipulus malus!
let Sextus=Cerebrum()
// =====================================================
// ================= TEST HARNESS ======================
// =====================================================

// ---------- utilities ----------
@inline(__always)
func sync() {
    stream.synchronize()
}

func mean(_ buf: GPUBuffer<Float>, _ count: Int) -> Float {
    let p = buf.ptr()
    var s: Float = 0
    for i in 0..<count { s += p[i] }
    return s / Float(count)
}

func variance(_ buf: GPUBuffer<Float>, _ count: Int) -> Float {
    let m = mean(buf, count)
    let p = buf.ptr()
    var s: Float = 0
    for i in 0..<count {
        let d = p[i] - m
        s += d * d
    }
    return s / Float(count)
}

func l1(_ buf: GPUBuffer<Float>, _ count: Int) -> Float {
    let p = buf.ptr()
    var s: Float = 0
    for i in 0..<count { s += abs(p[i]) }
    return s
}

func snapshot(_ buf: GPUBuffer<Float>, _ count: Int) -> [Float] {
    let p = buf.ptr()
    return (0..<count).map { p[$0] }
}

func cosine(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0
    var na: Float = 0
    var nb: Float = 0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    }
    return dot / sqrt(na * nb + 1e-9)
}

func delta(_ a: [Float], _ b: [Float]) -> Float {
    var s: Float = 0
    for i in 0..<a.count {
        s += abs(a[i] - b[i])
    }
    return s
}

// ---------- input ----------
let U = GPUBuffer<Float>(device: device, capacity: Int(Ds))
for i in 0..<Int(Ds) {
    U.ptr()[i] = Float.random(in: -1...1)
}

// warm-up
for _ in 0..<5 { Sextus.step(U_t: U) }
sync()

// =====================================================
// TEST 1: BG SELECTION (non-flat gating)
// =====================================================
print("\n=== TEST 1: BG SELECTION ===")

for i in 0..<Int(n) { Sextus.DA.ptr()[i] = 0.5 }
for _ in 0..<40 { Sextus.step(U_t: U) }
sync()

let gVar = variance(Sextus.BasalGanglia.g, Int(m))
print("BG g variance =", gVar)

// =====================================================
// TEST 2: DA POLICY SWITCH
// =====================================================
print("\n=== TEST 2: DA POLICY SWITCH ===")

for i in 0..<Int(n) { Sextus.DA.ptr()[i] = 0.0 }
for _ in 0..<40 { Sextus.step(U_t: U) }
sync()
let lowDA = snapshot(Sextus.Cortex.H_t0, Int(n*k))

for i in 0..<Int(n) { Sextus.DA.ptr()[i] = 1.0 }
for _ in 0..<40 { Sextus.step(U_t: U) }
sync()
let highDA = snapshot(Sextus.Cortex.H_t0, Int(n*k))

print("cos(lowDA, highDA) =", cosine(lowDA, highDA))

// =====================================================
// TEST 3: ACh ATTENTION / SPARSITY
// =====================================================
print("\n=== TEST 3: ACh ATTENTION ===")

for i in 0..<Int(n) { Sextus.ACh.ptr()[i] = 0.0 }
Sextus.step(U_t: U)
sync()
let lA0 = l1(Sextus.Cortex.H_t0, Int(n*k))

for i in 0..<Int(n) { Sextus.ACh.ptr()[i] = 1.0 }
Sextus.step(U_t: U)
sync()
let lA1 = l1(Sextus.Cortex.H_t0, Int(n*k))

print("L1(ACh=0) =", lA0)
print("L1(ACh=1) =", lA1)
print("sparsity ratio =", lA1 / max(lA0, 1e-6))

// =====================================================
// TEST 4: DA-GATED LEARNING (Cortex + BG)
// =====================================================
print("\n=== TEST 4: DA-GATED LEARNING ===")

let A0 = snapshot(Sextus.Cortex.A, Int(n*r))

for i in 0..<Int(n) { Sextus.DA.ptr()[i] = 0.0 }
Sextus.cortex_learn()
Sextus.bg_learn()
sync()
let A1 = snapshot(Sextus.Cortex.A, Int(n*r))

for i in 0..<Int(n) { Sextus.DA.ptr()[i] = 1.0 }
Sextus.cortex_learn()
Sextus.bg_learn()
sync()
let A2 = snapshot(Sextus.Cortex.A, Int(n*r))

print("ΔA low DA  =", delta(A0, A1))
print("ΔA high DA =", delta(A1, A2))

// =====================================================
// TEST 5: HISTORY DEPENDENCE
// =====================================================
print("\n=== TEST 5: HISTORY DEPENDENCE ===")

for i in 0..<Int(n) { Sextus.DA.ptr()[i] = 0.2 }
for _ in 0..<30 { Sextus.step(U_t: U) }
sync()
let histA = snapshot(Sextus.Cortex.H_t0, Int(n*k))

Sextus.Cortex.zero_state()
Sextus.Thalamus.zero_state()
sync()

for i in 0..<Int(n) { Sextus.DA.ptr()[i] = 0.8 }
for _ in 0..<30 { Sextus.step(U_t: U) }
sync()
let histB = snapshot(Sextus.Cortex.H_t0, Int(n*k))

print("cos(history A, history B) =", cosine(histA, histB))

print("\n=== ALL TESTS COMPLETE ===")