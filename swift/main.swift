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
        self.NE=GPUBuffer(device: device, capacity: Int(n))
        self.ACh=GPUBuffer(device: device, capacity: Int(n))
        self.d_NE=0.995
        self.d_ACh=0.998
        self.d_DA=0.997
        self.DA=GPUBuffer(device: device, capacity: Int(n))
        self.Cortex=cortex_setup()
        self.Thalamus=thalamus_setup()
        self.BasalGanglia=bg_setup()
        self.Cortex.zero_state()
        self.Thalamus.zero_state()
    }
    func chem_decay() {
        chem_decay_0(stream: stream, ACh, DA, NE, n, d_ACh, d_DA, d_NE)
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
        Thalamus.chem_mean(
            NE: NE,
            ACh: ACh,
            DA: DA
        )
        Cortex.learn(
            NE: NE,
            ACh: ACh,
            DA: DA,
            DA_c: Thalamus.DA_c
        )
    }
    func bg_learn(){
        Thalamus.chem_mean(
            NE: NE,
            ACh: ACh,
            DA: DA
        )
        BasalGanglia.learn(
            H_t0: Cortex.H_t0,
            DA_c: Thalamus.DA_c
        )
    }
}
//Sextus est discipulus malus!
let Sextus=Cerebrum()
// ============================================================
// FULL SELF-CONTAINED TEST HARNESS (WITH SENSORY NOISE)
// ============================================================

// ============================================================
// -------------------- SNAPSHOT HELPERS ----------------------
// ============================================================

@inline(__always)
func snapshot(_ buf: GPUBuffer<Float>, _ count: Int) -> [Float] {
    let p = buf.ptr()
    var out = [Float](repeating: 0, count: count)
    for i in 0..<count { out[i] = p[i] }
    return out
}

// ============================================================
// -------------------- MATH UTILITIES ------------------------
// ============================================================

func mean(_ x: [Float]) -> Float {
    x.reduce(0, +) / Float(x.count)
}

func variance(_ x: [Float]) -> Float {
    let m = mean(x)
    var v: Float = 0
    for xi in x {
        let d = xi - m
        v += d * d
    }
    return v / Float(x.count)
}

func l1(_ x: [Float]) -> Float {
    var s: Float = 0
    for xi in x { s += abs(xi) }
    return s
}

func cosine(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0
    var na: Float = 0
    var nb: Float = 0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        na  += a[i] * a[i]
        nb  += b[i] * b[i]
    }
    return dot / (sqrt(na * nb) + 1e-9)
}

func delta(_ a: [Float], _ b: [Float]) -> Float {
    var d: Float = 0
    for i in 0..<a.count {
        d += abs(a[i] - b[i])
    }
    return d
}

// ============================================================
// -------------------- NOISE INJECTION -----------------------
// ============================================================

let U_NOISE_SCALE: Float = 1e-3

@inline(__always)
func injectNoise(_ U: GPUBuffer<Float>, _ count: Int) {
    let p = U.ptr()
    for i in 0..<count {
        p[i] = Float.random(in: -1.0...1.0) * U_NOISE_SCALE
    }
}

// ============================================================
// -------------------- CONTROL HELPERS -----------------------
// ============================================================

func zeroAll(_ brain: Cerebrum) {
    brain.Cortex.zero_state()
    zero(stream: stream, brain.Cortex.elig, n*k, 1)
    zero(stream: stream, brain.BasalGanglia.g, m, 1)
    stream.synchronize()
}

func runSteps(_ brain: Cerebrum, U: GPUBuffer<Float>, steps: Int) {
    for _ in 0..<steps {
        injectNoise(U, Int(k))
        brain.step(U_t: U)
    }
    stream.synchronize()
}

// ============================================================
// -------------------- TEST INPUT ----------------------------
// ============================================================

let brain = Sextus
let U = GPUBuffer<Float>(device: device, capacity: Int(k))

// ============================================================
// TEST 1 — BG WINNER–TAKE–ALL
// ============================================================

print("""
============================================================
TEST 1: BASAL GANGLIA SELECTION
============================================================
""")

zeroAll(brain)
for i in 0..<Int(n) { brain.DA.ptr()[i] = 0.5 }
runSteps(brain, U: U, steps: 50)

let g = snapshot(brain.BasalGanglia.g, Int(m))
print("BG g variance =", variance(g))

// ============================================================
// TEST 2 — DA ATTRACTOR SHIFT
// ============================================================

print("""
============================================================
TEST 2: DOPAMINE ATTRACTOR SHIFT
============================================================
""")

zeroAll(brain)
for i in 0..<Int(n) { brain.DA.ptr()[i] = 0.0 }
runSteps(brain, U: U, steps: 60)
let H_low = snapshot(brain.Cortex.H_t0, Int(n*k))

zeroAll(brain)
for i in 0..<Int(n) { brain.DA.ptr()[i] = 1.0 }
runSteps(brain, U: U, steps: 60)
let H_high = snapshot(brain.Cortex.H_t0, Int(n*k))

print("cos(lowDA, highDA) =", cosine(H_low, H_high))

// ============================================================
// TEST 3 — ACh SUPPRESSION
// ============================================================

print("""
============================================================
TEST 3: ACETYLCHOLINE SUPPRESSION
============================================================
""")

zeroAll(brain)
for i in 0..<Int(n) { brain.ACh.ptr()[i] = 0.0 }
runSteps(brain, U: U, steps: 20)
let lA0 = l1(snapshot(brain.Cortex.H_t0, Int(n*k)))

zeroAll(brain)
for i in 0..<Int(n) { brain.ACh.ptr()[i] = 1.0 }
runSteps(brain, U: U, steps: 20)
let lA1 = l1(snapshot(brain.Cortex.H_t0, Int(n*k)))

print("L1(ACh=0) =", lA0)
print("L1(ACh=1) =", lA1)
print("ratio =", lA1 / max(lA0, 1e-9))

// ============================================================
// TEST 4 — DA-GATED LEARNING
// ============================================================

print("""
============================================================
TEST 4: DA-GATED LEARNING
============================================================
""")

zeroAll(brain)
runSteps(brain, U: U, steps: 30)
let A0 = snapshot(brain.Cortex.A, Int(n*r))

for i in 0..<Int(n) { brain.DA.ptr()[i] = 0.0 }
brain.cortex_learn()
stream.synchronize()
let A1 = snapshot(brain.Cortex.A, Int(n*r))

for i in 0..<Int(n) { brain.DA.ptr()[i] = 1.0 }
brain.cortex_learn()
stream.synchronize()
let A2 = snapshot(brain.Cortex.A, Int(n*r))

print("ΔA low DA  =", delta(A0, A1))
print("ΔA high DA =", delta(A1, A2))

// ============================================================
// TEST 5 — HISTORY DEPENDENCE
// ============================================================

print("""
============================================================
TEST 5: HISTORY DEPENDENCE
============================================================
""")

zeroAll(brain)
for i in 0..<Int(n) { brain.DA.ptr()[i] = 0.2 }
runSteps(brain, U: U, steps: 40)
let H_A = snapshot(brain.Cortex.H_t0, Int(n*k))

zeroAll(brain)
for i in 0..<Int(n) { brain.DA.ptr()[i] = 0.8 }
runSteps(brain, U: U, steps: 40)
let H_B = snapshot(brain.Cortex.H_t0, Int(n*k))

print("cos(history A, history B) =", cosine(H_A, H_B))

print("""
============================================================
ALL TESTS COMPLETE
============================================================
""")