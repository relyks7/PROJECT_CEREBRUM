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
let KR_maxDispatchesPerBuffer = 32
var KR_inflight: [MTLCommandBuffer] = []
var KR_currentCommandBuffer: MTLCommandBuffer?
var KR_currentEncoder: MTLComputeCommandEncoder?
var KR_dispatchesInCurrentCommand = 0
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
@inline(__always)
func KR_obtainEncoder() -> MTLComputeCommandEncoder {
    if let encoder = KR_currentEncoder {
        return encoder
    }
    KR_waitForFreeSlot()
    let cmd = KR_queue.makeCommandBuffer()!
    KR_currentCommandBuffer = cmd
    let encoder = cmd.makeComputeCommandEncoder()!
    KR_currentEncoder = encoder
    KR_dispatchesInCurrentCommand = 0
    return encoder
}
@inline(__always)
func KR_finishEncoderIfNeeded() {
    guard let encoder = KR_currentEncoder, let cmd = KR_currentCommandBuffer else { return }
    encoder.endEncoding()
    cmd.commit()
    KR_inflight.append(cmd)
    KR_currentEncoder = nil
    KR_currentCommandBuffer = nil
    KR_dispatchesInCurrentCommand = 0
}
public func kernel_runner_synchronize() {
    KR_finishEncoderIfNeeded()
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
final class BufferPool {
    private var buckets: [Int: [MTLBuffer]] = [:]
    private let lock = NSLock()

    private func bucket(for count: Int) -> Int {
        var size = 64
        var target = max(count, 1)
        while size < target {
            size <<= 1
        }
        return size
    }
    func acquire(minimumCapacity: Int) -> (MTLBuffer, Int) {
        if KR_device == nil {
            kernel_runner_init()
        }
        lock.lock()
        defer { lock.unlock() }
        let capacity = bucket(for: minimumCapacity)
        if var existing = buckets[capacity], !existing.isEmpty {
            let buffer = existing.removeLast()
            buckets[capacity] = existing
            return (buffer, capacity)
        }
        let buffer = KR_device.makeBuffer(length: capacity * MemoryLayout<Float>.size,
                                          options: .storageModeShared)!
        return (buffer, capacity)
    }
    func release(buffer: MTLBuffer, capacity: Int) {
        lock.lock()
        var existing = buckets[capacity] ?? []
        existing.append(buffer)
        buckets[capacity] = existing
        lock.unlock()
    }
}
public final class DeviceFloatBuffer {
    private enum Ownership {
        case owned
        case pooled(capacity: Int)
    }
    private static let pool = BufferPool()

    public let buffer: MTLBuffer
    public private(set) var count: Int
    public let capacity: Int
    private let ownership: Ownership

    public init(count: Int) {
        if KR_device == nil {
            kernel_runner_init()
        }
        self.count = count
        self.capacity = count
        self.buffer = KR_device.makeBuffer(length: count * MemoryLayout<Float>.size,
                                           options: .storageModeShared)!
        self.ownership = .owned
    }
    private init(buffer: MTLBuffer, count: Int, capacity: Int, ownership: Ownership) {
        self.buffer = buffer
        self.count = count
        self.capacity = capacity
        self.ownership = ownership
    }
    deinit {
        if case .pooled(let cap) = ownership {
            DeviceFloatBuffer.pool.release(buffer: buffer, capacity: cap)
        }
    }
    public static func temporary(count: Int) -> DeviceFloatBuffer {
        let (buffer, capacity) = pool.acquire(minimumCapacity: count)
        return DeviceFloatBuffer(buffer: buffer, count: count, capacity: capacity, ownership: .pooled(capacity: capacity))
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
    public func updateCount(_ newCount: Int) {
        precondition(newCount <= capacity, "New count exceeds capacity")
        count = newCount
    }
}
public func kernel_runner_call(
    _ kernelName: String,
    buffers: inout [KRBuffer],
    gridX: Int, gridY: Int, gridZ: Int,
    tgX: Int, tgY: Int, tgZ: Int
) {
    let pipe = KR_pipeline(kernelName)
    let enc = KR_obtainEncoder()
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
    KR_dispatchesInCurrentCommand += 1
    if KR_dispatchesInCurrentCommand >= KR_maxDispatchesPerBuffer {
        KR_finishEncoderIfNeeded()
    }
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
public func copy(
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
         "copy",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
}
public func fill(
    _ A: DeviceFloatBuffer,
    _ B: Float,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    var buffers: [KRBuffer]=[
        .buffer(A.buffer),
        .floatVal(B),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
        "fill",
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
        let out = nextN == 1 ? B : DeviceFloatBuffer.temporary(count: nextN * batch)
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
        let out = nextN == 1 ? B : DeviceFloatBuffer.temporary(count: nextN * batch)
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
        let out = nextN == 1 ? B : DeviceFloatBuffer.temporary(count: nextN * batch)
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
        let out = nextN == 1 ? B : DeviceFloatBuffer.temporary(count: nextN * batch)
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
    let H_raw_inter = DeviceFloatBuffer.temporary(count: Int(k*r))
    gemm1(H_t0, A, H_raw_inter, k, n, r, 1)
    let H_raw_ipt = DeviceFloatBuffer.temporary(count: Int(k*n))
    gemm2(H_raw_inter, B, H_raw_ipt, k, r, n, 1)
    let H_raw = DeviceFloatBuffer.temporary(count: Int(k*n))
    softlog(H_raw_ipt, H_raw, 1.7, k*n, 1)
    let mu = DeviceFloatBuffer.temporary(count: Int(n))
    mean_simd(H_raw, mu, k, n)
    let gamma = DeviceFloatBuffer.temporary(count: Int(n))
    abs_mean_simd(H_raw, gamma, k, n)
    let H_sub = DeviceFloatBuffer.temporary(count: Int(n*k))
    inhib_sub(H_raw, mu, H_sub, alpha_sub, n, k)
    inhib_div(H_sub, gamma, H_t1, alpha_div, eps, n, k)
}
public func fast_oja_correct(
    U: DeviceFloatBuffer,   // n × r
    V: DeviceFloatBuffer,   // n × r
    X: DeviceFloatBuffer,   // k × n  (H_t0)
    eta: Float,
    k: UInt32,
    n: UInt32,
    r: UInt32
){
    // ------------------------------------------
    // pU = X U      (k × r)
    // pV = X V      (k × r)
    // ------------------------------------------
    let pU = DeviceFloatBuffer.temporary(count: Int(k * r))
    let pV = DeviceFloatBuffer.temporary(count: Int(k * r))

    gemm1(X, U, pU, k, n, r, 1)
    gemm1(X, V, pV, k, n, r, 1)

    // ------------------------------------------
    // dU = Xᵀ pU    (n × r)
    // dV = Xᵀ pV    (n × r)
    // ------------------------------------------
    let dU = DeviceFloatBuffer.temporary(count: Int(n * r))
    let dV = DeviceFloatBuffer.temporary(count: Int(n * r))

    gemm3(X, pU, dU, n, k, r, 1)
    gemm3(X, pV, dV, n, k, r, 1)

    // scale by eta
    let etaArr = DeviceFloatBuffer.temporary(count: Int(n * r))
    etaArr.fill(eta)
    mul(dU, etaArr, dU, n, r)
    mul(dV, etaArr, dV, n, r)

    // U ← U + dU
    add(U, dU, U, n, r)

    // V ← V + dV
    add(V, dV, V, n, r)
}
public func slow_oja_correct(
    U: DeviceFloatBuffer,   // n × r
    V: DeviceFloatBuffer,   // n × r
    X: DeviceFloatBuffer,   // k × n
    eta: Float,
    k: UInt32,
    n: UInt32,
    r: UInt32
){
    // -------------------------------------------------
    // 1. pU = X U      (k × r)
    //    pV = X V      (k × r)
    // -------------------------------------------------
    let pU = DeviceFloatBuffer.temporary(count: Int(k * r))
    let pV = DeviceFloatBuffer.temporary(count: Int(k * r))

    gemm1(X, U, pU, k, n, r, 1)
    gemm1(X, V, pV, k, n, r, 1)

    // -------------------------------------------------
    // 2. A = Xᵀ pU    (n × r)
    //    B = Xᵀ pV    (n × r)
    // -------------------------------------------------
    let A = DeviceFloatBuffer.temporary(count: Int(n * r))
    let B = DeviceFloatBuffer.temporary(count: Int(n * r))

    gemm3(X, pU, A, n, k, r, 1)
    gemm3(X, pV, B, n, k, r, 1)

    // -------------------------------------------------
    // 3. Compute Gram matrices:
    //    Gu = pUᵀ pU   (r × r)
    //    Gv = pVᵀ pV   (r × r)
    // -------------------------------------------------
    let Gu = DeviceFloatBuffer.temporary(count: Int(r * r))
    let Gv = DeviceFloatBuffer.temporary(count: Int(r * r))

    gemm3(pU, pU, Gu, r, k, r, 1)
    gemm3(pV, pV, Gv, r, k, r, 1)

    // -------------------------------------------------
    // 4. UGu = U Gu    (n × r)
    //    VGv = V Gv    (n × r)
    // -------------------------------------------------
    let UGu = DeviceFloatBuffer.temporary(count: Int(n * r))
    let VGv = DeviceFloatBuffer.temporary(count: Int(n * r))

    gemm1(U, Gu, UGu, n, r, r, 1)
    gemm1(V, Gv, VGv, n, r, r, 1)

    // -------------------------------------------------
    // 5. dU = A - UGu
    //    dV = B - VGv
    // -------------------------------------------------
    let dU = DeviceFloatBuffer.temporary(count: Int(n * r))
    let dV = DeviceFloatBuffer.temporary(count: Int(n * r))

    sub(A, UGu, dU, n, r)
    sub(B, VGv, dV, n, r)

    // -------------------------------------------------
    // 6. Scale updates by η
    // -------------------------------------------------
    let etaArr = DeviceFloatBuffer.temporary(count: Int(n * r))
    etaArr.fill(eta)

    mul(dU, etaArr, dU, n, r)
    mul(dV, etaArr, dV, n, r)

    // -------------------------------------------------
    // 7. Apply updates
    // -------------------------------------------------
    add(U, dU, U, n, r)
    add(V, dV, V, n, r)
}
public func embedding_oja_update_correct(
    embeddingRow: DeviceFloatBuffer,   // e: n × 1
    error: DeviceFloatBuffer,          // x: n × 1
    eta: Float,
    n: UInt32
){
    precondition(embeddingRow.count == Int(n))
    precondition(error.count == Int(n))

    // ---------------------------------------------
    // 1. Compute dot = eᵀ x   (scalar)
    // ---------------------------------------------
    let prod = DeviceFloatBuffer.temporary(count: Int(n))
    mul(embeddingRow, error, prod, n, 1)   // prod[i] = e[i] * x[i]

    let dot = DeviceFloatBuffer(count: 1)
    sum_simd(prod, dot, n, 1)              // dot = Σ_i e[i] * x[i]

    // ---------------------------------------------
    // 2. Compute d * e
    // ---------------------------------------------
    let dot_broadcast = DeviceFloatBuffer.temporary(count: Int(n))
    dot_broadcast.fill(dot.toArray()[0])

    let scaled_e = DeviceFloatBuffer.temporary(count: Int(n))
    mul(embeddingRow, dot_broadcast, scaled_e, n, 1)  // scaled_e = d * e

    // ---------------------------------------------
    // 3. delta = x - scaled_e
    // ---------------------------------------------
    let delta = DeviceFloatBuffer.temporary(count: Int(n))
    sub(error, scaled_e, delta, n, 1)

    // ---------------------------------------------
    // 4. Δe = η * delta
    // ---------------------------------------------
    let eta_arr = DeviceFloatBuffer.temporary(count: Int(n))
    eta_arr.fill(eta)

    mul(delta, eta_arr, delta, n, 1)

    // ---------------------------------------------
    // 5. e ← e + Δe
    // ---------------------------------------------
    add(embeddingRow, delta, embeddingRow, n, 1)
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
    let token = DeviceFloatBuffer.temporary(count: Int(k))
    var idxArr: [UInt32] = [idx]
    embedding(
        embedding_matrix,
        idxArr,
        token,
        1,
        k,
        vocab_size
    )
    let expected = DeviceFloatBuffer.temporary(count: Int(k))
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
    let pred = DeviceFloatBuffer.temporary(count: Int(k))
    pred.copy(from: H_t1, sourceOffset: 0, targetOffset: 0, count: Int(k))
    let E = DeviceFloatBuffer.temporary(count: Int(k))
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
    let emb_row = DeviceFloatBuffer.temporary(count: Int(k))
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
public func benchmarkTrainingStep(
    sequenceLength: Int = 150_000,
    vocabSize: UInt32 = 267_735
) {
    print("---- Benchmarking step (seq=\(sequenceLength), vocab=\(vocabSize)) ----")
    kernel_runner_synchronize()
    let embeddingMatrix = DeviceFloatBuffer(count: Int(vocabSize) * Int(k))
    embeddingMatrix.fill(0)
    let tokenSeq = (0..<sequenceLength).map { _ in UInt32.random(in: 0..<vocabSize) }
    let targetSeq = (0..<sequenceLength).map { _ in UInt32.random(in: 0..<vocabSize) }

    let start = DispatchTime.now().uptimeNanoseconds
    for i in 0..<sequenceLength {
        use_up_token(
            H_t0: H_t0,
            H_t1: H_t1,
            expected_idx: targetSeq[i],
            embedding_matrix: embeddingMatrix,
            vocab_size: vocabSize,
            k: k,
            idx: tokenSeq[i],
            n: n,
            A: A,
            B: B,
            U: U,
            V: V,
            etanull: etanull,
            eta0: eta0,
            lambda: lambda,
            beta: beta
        )
    }
    kernel_runner_synchronize()
    let end = DispatchTime.now().uptimeNanoseconds
    let elapsed = Double(end - start) / 1_000_000_000.0
    let tokensPerSec = Double(sequenceLength) / max(elapsed, 1e-9)
    print(String(format: "Elapsed: %.2fs, throughput: %.1f tokens/s", elapsed, tokensPerSec))
    print("---------------------------------------------------------------")
}
kernel_runner_init()
benchmarkTrainingStep()
