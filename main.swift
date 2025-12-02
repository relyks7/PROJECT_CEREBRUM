import Metal
import Foundation
public enum KRBuffer {
    case floatArray([Float])
    case uint32Array([UInt32])
    case floatVal(Float)
    case uint32Val(UInt32)
    case buffer(MTLBuffer)
}
var KR_device: MTLDevice!
var KR_queue: MTLCommandQueue!
var KR_libraries: [MTLLibrary] = []
var KR_pipelines: [String : MTLComputePipelineState] = [:]
var KR_bufferCache: [ObjectIdentifier : MTLBuffer] = [:]
public func kernel_runner_init() {
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
func gpuBuffer(for array: inout [Float]) -> MTLBuffer {
    let key = ObjectIdentifier(array as AnyObject)
    let size = array.count * MemoryLayout<Float>.size

    if let existing = KR_bufferCache[key], existing.length >= size {
        memcpy(existing.contents(), &array, size)
        return existing
    }

    let buf = KR_device.makeBuffer(length: size, options: .storageModeShared)!
    memcpy(buf.contents(), &array, size)
    KR_bufferCache[key] = buf
    return buf
}
func gpuBuffer(for array: inout [UInt32]) -> MTLBuffer {
    let size = array.count * MemoryLayout<UInt32>.size
    let buf = KR_device.makeBuffer(length: size, options: .storageModeShared)!
    memcpy(buf.contents(), &array, size)
    return buf
}
public func kernel_runner_call(
    _ kernelName: String,
    buffers: inout [KRBuffer],
    gridX: Int, gridY: Int, gridZ: Int,
    tgX: Int, tgY: Int, tgZ: Int
) {
    let pipe = KR_pipeline(kernelName)

    let cmd = KR_queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!

    enc.setComputePipelineState(pipe)

    // Track Float-array buffers so we can read back after dispatch
    var floatArrayIndices: [Int] = []

    // ========================================================
    // BIND ALL ARGUMENTS
    // ========================================================
    for (i, arg) in buffers.enumerated() {
        switch arg {

        case .floatArray(let arr):
            var mutable = arr
            let buf = gpuBuffer(for: &mutable)
            enc.setBuffer(buf, offset: 0, index: i)
            floatArrayIndices.append(i)

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
    cmd.waitUntilCompleted()

    // ========================================================
    // COPY RESULTS BACK INTO Swift arrays
    // ========================================================
    for idx in floatArrayIndices {
        if case .floatArray(let oldArr) = buffers[idx] {
            let size = oldArr.count * MemoryLayout<Float>.size

            // Find the matching cached MTLBuffer
            let key = ObjectIdentifier(oldArr as AnyObject)
            guard let mtlBuf = KR_bufferCache[key] else { continue }

            var newArr = oldArr
            memcpy(&newArr, mtlBuf.contents(), size)
            buffers[idx] = .floatArray(newArr)
        }
    }
}
public func add(
    A: [Float],
    B: [Float],
    C: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
        kernelName: "add",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func div(
    A: [Float],
    B: [Float],
    C: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
        kernelName: "div",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func mul(
    A: [Float],
    B: [Float],
    C: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
        kernelName: "mul",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func sub(
    A: [Float],
    B: [Float],
    C: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
        kernelName: "sub",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func embedding(
    A: [Float],
    B: [UInt32],
    C: inout [Float],
    n: UInt32,
    d: UInt32,
    vocab_size: UInt32
){
    precondition(A.count == Int(vocab_size*d), "A has wrong size")
    precondition(B.count == Int(n), "B has wrong size")
    precondition(C.count == Int(n*d), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .uint32Array(B),
        .floatArray(C),
        .uint32Val(n),
        .uint32Val(d)
    ]
    kernel_runner_call(
        kernelName: "embedding",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: 1, gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func gemm1(
    A: [Float],
    B: [Float],
    C: inout [Float],
    m: UInt32,
    n: UInt32,
    p: UInt32,
    b: UInt32
){
    precondition(A.count == Int(m*n*b), "A has wrong size")
    precondition(B.count == Int(n*p*b), "B has wrong size")
    precondition(C.count == Int(m*p*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
        .uint32Val(m),
        .uint32Val(n),
        .uint32Val(p),
        .uint32Val(b)
    ]
    kernel_runner_call(
        kernelName: "gemm1",
        buffers: &buffers,
        gridX: ((Int(p)+31)/32), gridY: ((Int(m)+31)/32)*Int(b), gridZ:1,
        tgX: 32, tgY: 32, tgZ: 1
    )
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func gemm2(
    A: [Float],
    B: [Float],
    C: inout [Float],
    m: UInt32,
    n: UInt32,
    p: UInt32,
    b: UInt32
){
    precondition(A.count == Int(m*n*b), "A has wrong size")
    precondition(B.count == Int(n*p*b), "B has wrong size")
    precondition(C.count == Int(m*p*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
        .uint32Val(m),
        .uint32Val(n),
        .uint32Val(p),
        .uint32Val(b)
    ]
    kernel_runner_call(
        kernelName: "gemm2",
        buffers: &buffers,
        gridX: ((Int(p)+31)/32), gridY: ((Int(m)+31)/32)*Int(b), gridZ:1,
        tgX: 32, tgY: 32, tgZ: 1
    )
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func gemm3(
    A: [Float],
    B: [Float],
    C: inout [Float],
    m: UInt32,
    n: UInt32,
    p: UInt32,
    b: UInt32
){
    precondition(A.count == Int(m*n*b), "A has wrong size")
    precondition(B.count == Int(n*p*b), "B has wrong size")
    precondition(C.count == Int(m*p*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
        .uint32Val(m),
        .uint32Val(n),
        .uint32Val(p),
        .uint32Val(b)
    ]
    kernel_runner_call(
        kernelName: "gemm3",
        buffers: &buffers,
        gridX: ((Int(p)+31)/32), gridY: ((Int(m)+31)/32)*Int(b), gridZ:1,
        tgX: 32, tgY: 32, tgZ: 1
    )
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func layernorm(
    A: [Float],
    B: inout [Float],
    mu: [Float],
    sigma2: [Float],
    n: UInt32,
    eps: Float,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(mu.count == Int(b), "mu has wrong size")
    precondition(sigma2.count == Int(b), "sigma2 has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(mu),
        .floatArray(sigma2),
        .uint32Val(n),
        .floatVal(eps),
        .uint32Val(b)
    ]
    kernel_runner_call(
        kernelName: "layernorm",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedB) = buffers[1] { B = updatedB }
}
public func max_simd(
    A: [Float],
    B: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == ((Int(n)+127)/128)*Int(b), "B has wrong size")
    var cur=A
    var curN=n
    while curN>Int(b){
        let nextN=(curN+127)/128
        var out=[Float](repeating:0, count:nextN*Int(b))
        var buffers: [KRBuffer]=[
            .floatArray(cur),
            .floatArray(out),
            .uint32Val(curN),
            .uint32Val(b)
        ]
        kernel_runner_call(
            kernelName: "max_simd_reduce",
            buffers: &buffers,
            gridX: nextN, gridY: b, gridZ: 1,
            128, 1, 1
        );
        curN=nextN
        cur=out
    }
}
public func cortex_step(
    H_t0: [Float],
    W: [Float],
    H_t1: inout [Float],
    H_t1_out: inout [Float],
    k: UInt32,
    n: UInt32
){
    var gemm_buffers: [KRBuffer]= [
        .floatArray(H_t0),
        .floatArray(W),
        .floatArray(H_t1),
        .uint32Val(k),
        .uint32Val(n),
        .uint32Val(n),
        .uint32Val(1)
    ]
    kernel_runner_call(
    "gemm1",
    buffers: &gemm_buffers,
    gridX: 32, gridY: 32, gridZ: 1,
    tgX: 32, tgY: 32, tgZ: 1
    )
    if case .floatArray(let updated) = gemm_buffers[2] {
        H_t1 = updated
    }
    var tanh_buffers: [KRBuffer]= [
        .floatArray(H_t1),
        .floatArray(H_t1_out),
        .uint32Val(n*k),
        .uint32Val(1)
    ]
    kernel_runner_call(
    "tanh",
    buffers: &tanh_buffers,
    gridX: (n*k+255)/256, gridY: 1, gridZ: 1,
    tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updated2) = tanh_buffers[1] {
        H_t1_out = updated2
    }
}
public func fast_oja(
    X: [Float],
    Y: [Float],
    U: [Float],
    V: [Float],
    P: [Float],
    Q: [Float],
    Ysq: inout [Float],
    nj: inout [Float],
    etanull: Float,
    k: UInt32,
    n: UInt32
){
  var sq_buffers: [KRBuffer]=[
    .floatArray(Y),
    .floatArray(Ysq),
    .uint32Val(n*k),
    .uint32Val(1)
  ]
  kernel_runner_call(
    "square",
    buffers=&sq_buffers,
    gridX: (n*k+255)/256, gridY: 1, gridZ: 1,
    tgX: 256, tgY: 1, tgZ: 1
  )
  var sum_buffers1: [KRBuffer]=[
    .floatArray(Ysq),
    .floatArray(nj),
    .uint32Val(n),
    .uint32Val(k)
  ]
  kernel_runner_call(
    "sum_simd_reduce",
    buffers=&sq_buffers,
    gridX: ((n+127)/128), gridY: k, gridZ: 1,
    tgX: 128, tgY: 1, tgZ: 1
  )
  var sum_buffers2: [KRBuffer]=[
    .floatArray(Ysq),
    .floatArray(nj),
    .uint32Val((n+127)/128),
    .uint32Val(k)
  ]
  kernel_runner_call(
    "sum_simd_reduce",
    buffers=&sq_buffers,
    gridX: ((n+127)/128), gridY: ((k+127)/128), gridZ: 1,
    tgX: 128, tgY: 1, tgZ: 1
  )
}
kernel_runner_init()