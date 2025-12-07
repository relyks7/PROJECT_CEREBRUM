import Metal
import Foundation
import Darwin
public enum KRBuffer {
    case uint32Array([UInt32])
    case floatVal(Float)
    case uint32Val(UInt32)
    case buffer(MTLBuffer)
}
var KR_device: MTLDevice!
var KR_queue: MTLCommandQueue!
var KR_libraries: [MTLLibrary] = []
var KR_pipelines: [String : MTLComputePipelineState] = [:]
let KR_maxInflightBuffers = 3
var KR_inflight: [MTLCommandBuffer] = []
public func kernel_runner_init() {
    if KR_device != nil { return }
    KR_device = MTLCreateSystemDefaultDevice()!
    KR_queue  = KR_device.makeCommandQueue()!

    print("Metal device: \(KR_device.name)")

    let fm = FileManager.default
    let kernelsURL = URL(fileURLWithPath: "./kernels")

    guard let items = try? fm.contentsOfDirectory(at: kernelsURL,
                                                  includingPropertiesForKeys: nil,
                                                  options: [])
    else {
        fatalError("❌ Could not read ./kernels directory")
    }

    for file in items {
        if file.pathExtension == "metallib" {
            do {
                let lib = try KR_device.makeLibrary(URL: file)
                KR_libraries.append(lib)
                print("Loaded library: \(file.lastPathComponent)")
            } catch {
                print("⚠️ Failed to load \(file.lastPathComponent): \(error)")
            }
        }
    }

    if KR_libraries.isEmpty {
        fatalError("❌ No .metallib files found in ./kernels/")
    }

    print("Kernel runner initialized.\n")
}
@inline(__always)
func KR_waitForFreeSlot() {
    if KR_inflight.count >= KR_maxInflightBuffers {
        let oldest = KR_inflight.removeFirst()
        oldest.waitUntilCompleted()
    }
}
public func kernel_runner_synchronize() {
    while !KR_inflight.isEmpty {
        let pending = KR_inflight.removeFirst()
        pending.waitUntilCompleted()
    }
}
func KR_pipeline(_ name: String) -> MTLComputePipelineState {
    if let cached = KR_pipelines[name] {
        return cached
    }

    for lib in KR_libraries {
        if let fn = lib.makeFunction(name: name) {
            let pipe = try! KR_device.makeComputePipelineState(function: fn)
            KR_pipelines[name] = pipe
            return pipe
        }
    }

    fatalError("❌ Kernel '\(name)' not found in ANY metallib")
}
func gpuBuffer(for array: inout [UInt32]) -> MTLBuffer {
    let size = array.count * MemoryLayout<UInt32>.size
    let buf = KR_device.makeBuffer(length: size, options: .storageModeShared)!
    memcpy(buf.contents(), &array, size)
    return buf
}
public final class DeviceFloatBuffer {
    public let buffer: MTLBuffer
    public let count: Int

    public init(count: Int) {
        if KR_device == nil {
            kernel_runner_init()
        }
        self.count = count
        self.buffer = KR_device.makeBuffer(length: count * MemoryLayout<Float>.size,
                                           options: .storageModeShared)!
    }
    public convenience init(_ host: [Float]) {
        self.init(count: host.count)
        copy(from: host)
    }
    @inline(__always)
    private func basePointer() -> UnsafeMutablePointer<Float> {
        buffer.contents().assumingMemoryBound(to: Float.self)
    }
    public func fill(_ value: Float) {
        kernel_runner_synchronize()
        let ptr = basePointer()
        for i in 0..<count {
            ptr[i] = value
        }
    }
    public func copy(from host: [Float], targetOffset: Int = 0) {
        kernel_runner_synchronize()
        precondition(host.count + targetOffset <= count, "Host data exceeds buffer")
        host.withUnsafeBytes { bytes in
            let dst = basePointer().advanced(by: targetOffset)
            memcpy(dst, bytes.baseAddress!, host.count * MemoryLayout<Float>.size)
        }
    }
    public func toArray(range: Range<Int>? = nil) -> [Float] {
        kernel_runner_synchronize()
        let r = range ?? 0..<count
        precondition(r.lowerBound >= 0 && r.upperBound <= count, "Range out of bounds")
        var out = [Float](repeating: 0, count: r.count)
        memcpy(&out, basePointer().advanced(by: r.lowerBound), r.count * MemoryLayout<Float>.size)
        return out
    }
    public func copy(from other: DeviceFloatBuffer,
                     sourceOffset: Int = 0,
                     targetOffset: Int = 0,
                     count elements: Int? = nil) {
        kernel_runner_synchronize()
        let copyCount = elements ?? min(other.count - sourceOffset, count - targetOffset)
        precondition(copyCount >= 0, "Invalid copy count")
        precondition(sourceOffset + copyCount <= other.count, "Source out of range")
        precondition(targetOffset + copyCount <= count, "Destination out of range")
        memcpy(basePointer().advanced(by: targetOffset),
               other.basePointer().advanced(by: sourceOffset),
               copyCount * MemoryLayout<Float>.size)
    }
}
public func kernel_runner_call(
    _ kernelName: String,
    buffers: inout [KRBuffer],
    gridX: Int, gridY: Int, gridZ: Int,
    tgX: Int, tgY: Int, tgZ: Int
) {
    KR_waitForFreeSlot()
    let pipe = KR_pipeline(kernelName)

    let cmd = KR_queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!

    enc.setComputePipelineState(pipe)

    // ========================================================
    // BIND ALL ARGUMENTS
    // ========================================================
    for (i, arg) in buffers.enumerated() {
        switch arg {

        case .uint32Array(let arr):
            var temp = arr
            let buf = gpuBuffer(for: &temp)
            enc.setBuffer(buf, offset: 0, index: i)

        case .floatVal(var value):
            enc.setBytes(&value, length: MemoryLayout<Float>.size, index: i)

        case .uint32Val(var value):
            enc.setBytes(&value, length: MemoryLayout<UInt32>.size, index: i)

        case .buffer(let buf):
            enc.setBuffer(buf, offset: 0, index: i)
        }
    }

    // ========================================================
    // DISPATCH
    // ========================================================
    enc.dispatchThreadgroups(
        MTLSize(width: gridX, height: gridY, depth: gridZ),
        threadsPerThreadgroup: MTLSize(width: tgX, height: tgY, depth: tgZ)
    )

    enc.endEncoding()
    cmd.commit()
    KR_inflight.append(cmd)
}
public func add(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "add",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func add_unsafe(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "add",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func div(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "div",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func mul(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "mul",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func sub(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "sub",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func inhib_sub(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ alpha: Float,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .floatVal(alpha),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
        "inhib_sub",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func inhib_div(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ alpha: Float,
    _ eps: Float,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .floatVal(alpha),
        .floatVal(eps),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
        "inhib_div",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func embedding(
    _ A: DeviceFloatBuffer,
    _ B: [UInt32],
    _ C: DeviceFloatBuffer,
    _ n: UInt32,
    _ d: UInt32,
    _ vocab_size: UInt32
){
    precondition(A.count == Int(vocab_size*d), "A has wrong size")
    precondition(C.count == Int(n*d), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .uint32Array(B),
        .buffer(C.buffer),
        .uint32Val(n),
        .uint32Val(d)
    ]
    kernel_runner_call(
         "embedding",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: 1, gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func gemm1(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ m: UInt32,
    _ n: UInt32,
    _ p: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(m*n*b), "A has wrong size")
    precondition(B.count == Int(n*p*b), "B has wrong size")
    precondition(C.count == Int(m*p*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .uint32Val(m),
        .uint32Val(n),
        .uint32Val(p),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "gemm1",
        buffers: &buffers,
        gridX: ((Int(p)+31)/32), gridY: ((Int(m)+31)/32)*Int(b), gridZ:1,
        tgX: 32, tgY: 32, tgZ: 1
    )
}
public func gemm2(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ m: UInt32,
    _ n: UInt32,
    _ p: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(m*n*b), "A has wrong size")
    precondition(B.count == Int(n*p*b), "B has wrong size")
    precondition(C.count == Int(m*p*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .uint32Val(m),
        .uint32Val(n),
        .uint32Val(p),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "gemm2",
        buffers: &buffers,
        gridX: ((Int(p)+31)/32), gridY: ((Int(m)+31)/32)*Int(b), gridZ:1,
        tgX: 32, tgY: 32, tgZ: 1
    )
}
public func gemm3(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ m: UInt32,
    _ n: UInt32,
    _ p: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(m*n*b), "A has wrong size")
    precondition(B.count == Int(n*p*b), "B has wrong size")
    precondition(C.count == Int(m*p*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(C.buffer),
        .uint32Val(m),
        .uint32Val(n),
        .uint32Val(p),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "gemm3",
        buffers: &buffers,
        gridX: ((Int(p)+31)/32), gridY: ((Int(m)+31)/32)*Int(b), gridZ:1,
        tgX: 32, tgY: 32, tgZ: 1
    )
}
public func layernorm(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ mu: DeviceFloatBuffer,
    _ sigma2: DeviceFloatBuffer,
    _ n: UInt32,
    _ eps: Float,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(mu.count == Int(b), "mu has wrong size")
    precondition(sigma2.count == Int(b), "sigma2 has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(mu.buffer),
        .buffer(sigma2.buffer),
        .uint32Val(n),
        .floatVal(eps),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "layernorm",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func max_simd(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(b), "B has wrong size")
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out = nextN == 1 ? B : DeviceFloatBuffer(count: nextN * batch)
        var buffers: [KRBuffer]=[
            .buffer(cur.buffer),
            .buffer(out.buffer),
            .uint32Val(UInt32(curN)),
            .uint32Val(b)
        ]
        kernel_runner_call(
             "max_simd_reduce",
            buffers: &buffers,
            gridX: nextN, gridY: batch, gridZ: 1,
            tgX: 128, tgY: 1, tgZ: 1
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
    if cur !== B {
        B.copy(from: cur, count: B.count)
    }
}
public func sum_simd(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(b), "B has wrong size")
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out = nextN == 1 ? B : DeviceFloatBuffer(count: nextN * batch)
        var buffers: [KRBuffer]=[
            .buffer(cur.buffer),
            .buffer(out.buffer),
            .uint32Val(UInt32(curN)),
            .uint32Val(b)
        ]
        kernel_runner_call(
             "sum_simd_reduce",
            buffers: &buffers,
            gridX: nextN, gridY: batch, gridZ: 1,
            tgX: 128, tgY: 1, tgZ: 1
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
    if cur !== B {
        B.copy(from: cur, count: B.count)
    }
}
public func mean_simd(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(b), "B has wrong size")
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    var firstPass = true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out = nextN == 1 ? B : DeviceFloatBuffer(count: nextN * batch)
        var buffers: [KRBuffer]=[
            .buffer(cur.buffer),
            .buffer(out.buffer),
            .uint32Val(UInt32(curN)),
            .uint32Val(b)
        ]
        let kernelName = firstPass ? "mean_simd_reduce" : "sum_simd_reduce"
        kernel_runner_call(
             kernelName,
            buffers: &buffers,
            gridX: nextN, gridY: batch, gridZ: 1,
            tgX: 128, tgY: 1, tgZ: 1
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
        firstPass = false
    }
    if cur !== B {
        B.copy(from: cur, count: B.count)
    }
}
public func abs_mean_simd(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(b), "B has wrong size")
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    var firstPass = true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out = nextN == 1 ? B : DeviceFloatBuffer(count: nextN * batch)
        var buffers: [KRBuffer]=[
            .buffer(cur.buffer),
            .buffer(out.buffer),
            .uint32Val(UInt32(curN)),
            .uint32Val(b)
        ]
        let kernelName = firstPass ? "abs_mean_simd_reduce" : "sum_simd_reduce"
        kernel_runner_call(
             kernelName,
            buffers: &buffers,
            gridX: nextN, gridY: batch, gridZ: 1,
            tgX: 128, tgY: 1, tgZ: 1
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
        firstPass = false
    }
    if cur !== B {
        B.copy(from: cur, count: B.count)
    }
}
public func tanh(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "tanh",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func softlog(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ alpha: Float,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .uint32Val(n),
        .floatVal(alpha),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "softlog",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func relu(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "relu",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func softmax(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ global_max: DeviceFloatBuffer,
    _ denom: DeviceFloatBuffer,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(global_max.count == Int(b), "global_max has wrong size")
    precondition(denom.count == Int(b), "denom has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .buffer(B.buffer),
        .buffer(global_max.buffer),
        .buffer(denom.buffer),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "softmax",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func outer_prod(
    _ A: DeviceFloatBuffer,
    _ B: DeviceFloatBuffer,
    _ C: DeviceFloatBuffer,
    _ n: UInt32,
    _ m: UInt32,
    _ b: UInt32
){
    gemm1(
        A, B, C,
        n, 1, m,
        b
    )
}
public func cortex_step(
    H_t0: DeviceFloatBuffer,
    A: DeviceFloatBuffer,
    B: DeviceFloatBuffer,
    H_t1: DeviceFloatBuffer,
    alpha_sub: Float,
    alpha_div: Float,
    k: UInt32,
    r: UInt32,
    n: UInt32
){
    let H_raw_inter = DeviceFloatBuffer(count: Int(k*r))
    gemm1(H_t0, A, H_raw_inter, k, n, r, 1)
    let H_raw_ipt = DeviceFloatBuffer(count: Int(k*n))
    gemm2(H_raw_inter, B, H_raw_ipt, k, r, n, 1)
    let H_raw = DeviceFloatBuffer(count: Int(k*n))
    softlog(H_raw_ipt, H_raw, 1.7, k*n, 1)
    let mu = DeviceFloatBuffer(count: Int(n))
    mean_simd(H_raw, mu, k, n)
    let gamma = DeviceFloatBuffer(count: Int(n))
    abs_mean_simd(H_raw, gamma, k, n)
    let H_sub = DeviceFloatBuffer(count: Int(n*k))
    inhib_sub(H_raw, mu, H_sub, alpha_sub, n, k)
    inhib_div(H_sub, gamma, H_t1, alpha_div, eps, n, k)
}
public func fast_oja(
    A: DeviceFloatBuffer,
    B: DeviceFloatBuffer,
    U: DeviceFloatBuffer,
    V: DeviceFloatBuffer,
    X: DeviceFloatBuffer,
    Y: DeviceFloatBuffer,
    etanull: Float,
    k: UInt32,
    n: UInt32,
    r: UInt32
){
    // ---------------------------
    // 1. softlog(U), softlog(V)
    // ---------------------------
    let U_t = DeviceFloatBuffer(count: Int(n*r))
    let V_t = DeviceFloatBuffer(count: Int(n*r))
    softlog(U, U_t, 1.7, n, r)
    softlog(V, V_t, 1.7, n, r)

    // M = etanull * (U_t ⊙ V_t)
    let M = DeviceFloatBuffer(count: Int(n*r))
    mul(U_t, V_t, M, n, r)

    let eta_arr = DeviceFloatBuffer(count: Int(n*r))
    eta_arr.fill(etanull)
    mul(M, eta_arr, M, n, r)

    // ---------------------------
    // 2. p = X B    (k×n → k×r)
    // ---------------------------
    let p = DeviceFloatBuffer(count: Int(k*r))
    gemm1(X, B, p, k, n, r, 1)

    // ---------------------------
    // 3. q = Y A    (k×n → k×r)
    // ---------------------------
    let q = DeviceFloatBuffer(count: Int(k*r))
    gemm1(Y, A, q, k, n, r, 1)

    // ---------------------------
    // 4. Yᵀ p , Yᵀ q   (n×k → n×r)
    // ---------------------------
    let YTp = DeviceFloatBuffer(count: Int(n*r))
    let YTq = DeviceFloatBuffer(count: Int(n*r))
    gemm3(Y, p, YTp, n, k, r, 1)
    gemm3(Y, q, YTq, n, k, r, 1)

    // ---------------------------
    // 5. G = Yᵀp − Yᵀq
    // ---------------------------
    let G = DeviceFloatBuffer(count: Int(n*r))
    sub(YTp, YTq, G, n, r)

    // ---------------------------
    // 6. ΔA = M ⊙ G
    // ---------------------------
    let dA = DeviceFloatBuffer(count: Int(n*r))
    mul(M, G, dA, n, r)

    // ---------------------------
    // 7. ΔB = M ⊙ G
    // ---------------------------
    let dB = DeviceFloatBuffer(count: Int(n*r))
    mul(M, G, dB, n, r)

    // ---------------------------
    // 8. Update A and B
    // ---------------------------
    add(A, dA, A, n, r)
    add(B, dB, B, n, r)
}
public func slow_oja(
    U: DeviceFloatBuffer,
    V: DeviceFloatBuffer,
    A: DeviceFloatBuffer,
    B: DeviceFloatBuffer,
    X: DeviceFloatBuffer,
    Y: DeviceFloatBuffer,
    E: DeviceFloatBuffer,      // shape: k
    eta0: Float,
    lambda: Float,
    beta: Float,
    n: UInt32,
    r: UInt32,
    k: UInt32
){
    // ---------------------------
    // 0. Global neuromodulatory factor δ
    // O(k) CPU — negligible
    // ---------------------------
    var absSum: Float = 0
    let err = E.toArray()
    for i in 0..<Int(k) { absSum += abs(err[i]) }

    let deltaRaw = absSum / Float(k)
    let delta = tanh(beta * deltaRaw)

    // ---------------------------
    // 1. softlog(U), softlog(V)
    // ---------------------------
    let U_t = DeviceFloatBuffer(count: Int(n*r))
    let V_t = DeviceFloatBuffer(count: Int(n*r))
    softlog(U, U_t, 1.7, n, r)
    softlog(V, V_t, 1.7, n, r)

    // ---------------------------
    // 2. M = eta0 * (U_t ⊙ V_t)
    // ---------------------------
    let M = DeviceFloatBuffer(count: Int(n*r))
    mul(U_t, V_t, M, n, r)

    let etaArr = DeviceFloatBuffer(count: Int(n*r))
    etaArr.fill(eta0)
    mul(M, etaArr, M, n, r)

    // ---------------------------
    // 3. G = Yᵀ(XB) − Yᵀ(YA)
    // ---------------------------
    let p = DeviceFloatBuffer(count: Int(k*r))
    gemm1(X, B, p, k, n, r, 1)

    let q = DeviceFloatBuffer(count: Int(k*r))
    gemm1(Y, A, q, k, n, r, 1)

    let YTp = DeviceFloatBuffer(count: Int(n*r))
    let YTq = DeviceFloatBuffer(count: Int(n*r))
    gemm3(Y, p, YTp, n, k, r, 1)
    gemm3(Y, q, YTq, n, k, r, 1)

    let G = DeviceFloatBuffer(count: Int(n*r))
    sub(YTp, YTq, G, n, r)

    // ---------------------------
    // 4. H_fast = M ⊙ G
    // ---------------------------
    let H_fast = DeviceFloatBuffer(count: Int(n*r))
    mul(M, G, H_fast, n, r)

    // ---------------------------
    // 5. S = δ * (H_fast ⊙ G)
    // ---------------------------
    let S = DeviceFloatBuffer(count: Int(n*r))
    mul(H_fast, G, S, n, r)

    if delta != 0 {
        let deltaArr = DeviceFloatBuffer(count: Int(n*r))
        deltaArr.fill(delta)
        mul(S, deltaArr, S, n, r)
    }

    // ---------------------------
    // 6. ΔU = λ * S V
    // ---------------------------
    let dU = DeviceFloatBuffer(count: Int(n*r))
    gemm1(S, V, dU, n, r, r, 1)

    // ---------------------------
    // 7. ΔV = λ * Sᵀ U
    // ---------------------------
    let dV = DeviceFloatBuffer(count: Int(n*r))
    gemm3(S, U, dV, n, r, r, 1)

    let lamArr = DeviceFloatBuffer(count: Int(n*r))
    lamArr.fill(lambda)
    mul(dU, lamArr, dU, n, r)
    mul(dV, lamArr, dV, n, r)

    // ---------------------------
    // 8. Update U and V
    // ---------------------------
    add(U, dU, U, n, r)
    add(V, dV, V, n, r)
}
public func embedding_oja_update(
    embeddingRow: DeviceFloatBuffer,
    error: DeviceFloatBuffer,
    eta: Float,
    alpha: Float,
    n: UInt32
){
    precondition(embeddingRow.count == Int(n))
    precondition(error.count == Int(n))

    // -------------------------------------------------
    // 1. err ⊙ e
    // -------------------------------------------------
    let hebb = DeviceFloatBuffer(count: Int(n))
    mul(error, embeddingRow, hebb, n, 1)

    // -------------------------------------------------
    // 2. e ⊙ e  (normalization term)
    // -------------------------------------------------
    let ee = DeviceFloatBuffer(count: Int(n))
    mul(embeddingRow, embeddingRow, ee, n, 1)

    // -------------------------------------------------
    // 3. G = hebb − α * ee
    // -------------------------------------------------
    let scaled_ee = DeviceFloatBuffer(count: Int(n))
    scaled_ee.fill(alpha)
    mul(ee, scaled_ee, scaled_ee, n, 1)

    let G = DeviceFloatBuffer(count: Int(n))
    sub(hebb, scaled_ee, G, n, 1)

    // -------------------------------------------------
    // 4. Δe = η * G
    // -------------------------------------------------
    let eta_arr = DeviceFloatBuffer(count: Int(n))
    eta_arr.fill(eta)
    let dE = DeviceFloatBuffer(count: Int(n))
    mul(G, eta_arr, dE, n, 1)

    // -------------------------------------------------
    // 5. e ← e + Δe
    // -------------------------------------------------
    add(embeddingRow, dE, embeddingRow, n, 1)
}
let alpha_s: Float=0.05
let alpha_d: Float=0.10
let eps: Float=1e-6
let steps: Int=3
let etanull: Float=0.003
let eta0: Float=0.0003
let lambda: Float=0.0001
let beta: Float=3.0
let k: UInt32=512
let n: UInt32=1024
let r: UInt32=32
let vocab_size: UInt32=1048576
func randomArray(count: Int, range: ClosedRange<Float>) -> [Float] {
    return (0..<count).map { _ in Float.random(in: range) }
}
var A = DeviceFloatBuffer(randomArray(count: Int(n)*Int(r), range: -0.01...0.01))
var B = DeviceFloatBuffer(randomArray(count: Int(n)*Int(r), range: -0.01...0.01))
var U = DeviceFloatBuffer(randomArray(count: Int(n)*Int(r), range: -0.1...0.1))
var V = DeviceFloatBuffer(randomArray(count: Int(n)*Int(r), range: -0.1...0.1))
var H_t0 = DeviceFloatBuffer(count: Int(k*n))
var H_t1 = DeviceFloatBuffer(count: Int(k*n))
H_t0.fill(0)
H_t1.fill(0)
public func zerostate(
    H_t0: DeviceFloatBuffer,
    H_t1: DeviceFloatBuffer,
    k: UInt32,
    n: UInt32
){
    let size = Int(k * n)
    precondition(H_t0.count == size)
    precondition(H_t1.count == size)
    H_t0.fill(0)
    H_t1.fill(0)
}
public func use_up_token(
    H_t0: DeviceFloatBuffer,
    H_t1: DeviceFloatBuffer,
    expected_idx: UInt32,
    embedding_matrix: DeviceFloatBuffer,
    vocab_size: UInt32,
    k: UInt32,
    idx: UInt32,
    n: UInt32,
    A: DeviceFloatBuffer,
    B: DeviceFloatBuffer,
    U: DeviceFloatBuffer,
    V: DeviceFloatBuffer,
    etanull: Float,
    eta0: Float,
    lambda: Float,
    beta: Float
){
    let token = DeviceFloatBuffer(count: Int(k))
    var idxArr: [UInt32] = [idx]
    embedding(
        embedding_matrix,
        idxArr,
        token,
        1,
        k,
        vocab_size
    )
    let expected = DeviceFloatBuffer(count: Int(k))
    idxArr = [expected_idx]
    embedding(
        embedding_matrix,
        idxArr,
        expected,
        1,
        k,
        vocab_size
    )
    add_unsafe(
        H_t0,
        token,
        H_t0,
        k,
        1
    )
    cortex_step(
        H_t0: H_t0,
        A: A,
        B: B,
        H_t1: H_t1,
        alpha_sub: alpha_s,
        alpha_div: alpha_d,
        k: k,
        r: r,
        n: n
    )
    let pred = DeviceFloatBuffer(count: Int(k))
    pred.copy(from: H_t1, sourceOffset: 0, targetOffset: 0, count: Int(k))
    let E = DeviceFloatBuffer(count: Int(k))
    sub(expected, pred, E, k, 1)
    fast_oja(
        A: A,
        B: B,
        U: U,
        V: V,
        X: H_t0,
        Y: H_t1,
        etanull: etanull,
        k: k,
        n: n,
        r: r
    )
    slow_oja(
        U: U,
        V: V,
        A: A,
        B: B,
        X: H_t0,
        Y: H_t1,
        E: E,
        eta0: eta0,
        lambda: lambda,
        beta: beta,
        n: n,
        r: r,
        k: k
    )
    let base = Int(idx) * Int(k)
    let emb_row = DeviceFloatBuffer(count: Int(k))
    emb_row.copy(from: embedding_matrix, sourceOffset: base, targetOffset: 0, count: Int(k))
    embedding_oja_update(
        embeddingRow: emb_row,
        error: E,
        eta: 0.0005,
        alpha: 0.05,
        n: k
    )
    embedding_matrix.copy(from: emb_row, sourceOffset: 0, targetOffset: base, count: Int(k))
    H_t0.copy(from: H_t1)
}
kernel_runner_init()
