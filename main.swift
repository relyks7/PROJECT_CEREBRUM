import Metal
import Foundation
import Darwin
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
    let size = array.count * MemoryLayout<Float>.size
    let buf = KR_device.makeBuffer(length: size, options: .storageModeShared)!
    memcpy(buf.contents(), &array, size)
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

    // Track Float-array buffers and their bound MTLBuffers for readback
    var floatBindings: [(index: Int, buffer: MTLBuffer, count: Int)] = []

    // ========================================================
    // BIND ALL ARGUMENTS
    // ========================================================
    for (i, arg) in buffers.enumerated() {
        switch arg {

        case .floatArray(let arr):
            var mutable = arr
            let buf = gpuBuffer(for: &mutable)
            enc.setBuffer(buf, offset: 0, index: i)
            floatBindings.append((i, buf, arr.count))

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
    for binding in floatBindings {
        if case .floatArray(let oldArr) = buffers[binding.index] {
            var newArr = oldArr
            let size = min(oldArr.count, binding.count) * MemoryLayout<Float>.size
            memcpy(&newArr, binding.buffer.contents(), size)
            buffers[binding.index] = .floatArray(newArr)
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
         "add",
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
         "div",
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
         "mul",
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
         "sub",
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
         "embedding",
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
         "gemm1",
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
         "gemm2",
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
         "gemm3",
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
         "layernorm",
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
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    while curN > batch {
        let nextN = (curN + 127) / 128
        var out = [Float](repeating: 0, count: nextN * batch)
        var buffers: [KRBuffer]=[
            .floatArray(cur),
            .floatArray(out),
            .uint32Val(UInt32(curN)),
            .uint32Val(b)
        ]
        kernel_runner_call(
             "max_simd_reduce",
            buffers: &buffers,
            gridX: nextN, gridY: batch, gridZ: 1,
            tgX: 128, tgY: 1, tgZ: 1
        );
        if case .floatArray(let updatedOut) = buffers[1] {
            out = updatedOut
        }
        cur = out
        curN = nextN
    }
    B = cur
}
public func sum_simd(
    A: [Float],
    B: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == ((Int(n)+127)/128)*Int(b), "B has wrong size")
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    while curN > batch {
        let nextN = (curN + 127) / 128
        var out = [Float](repeating: 0, count: nextN * batch)
        var buffers: [KRBuffer]=[
            .floatArray(cur),
            .floatArray(out),
            .uint32Val(UInt32(curN)),
            .uint32Val(b)
        ]
        kernel_runner_call(
             "sum_simd_reduce",
            buffers: &buffers,
            gridX: nextN, gridY: batch, gridZ: 1,
            tgX: 128, tgY: 1, tgZ: 1
        );
        if case .floatArray(let updatedOut) = buffers[1] {
            out = updatedOut
        }
        cur = out
        curN = nextN
    }
    B = cur
}
public func mean_simd(
    A: [Float],
    B: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == ((Int(n)+127)/128)*Int(b), "B has wrong size")
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    var firstPass = true
    while curN > batch {
        let nextN = (curN + 127) / 128
        var out = [Float](repeating: 0, count: nextN * batch)
        var buffers: [KRBuffer]=[
            .floatArray(cur),
            .floatArray(out),
            .uint32Val(UInt32(curN)),
            .uint32Val(b)
        ]
        let kernelName = firstPass ? "mean_simd_reduce" : "sum_simd_reduce"
        kernel_runner_call(
             kernelName,
            buffers: &buffers,
            gridX: nextN, gridY: batch, gridZ: 1,
            tgX: 128, tgY: 1, tgZ: 1
        );
        if case .floatArray(let updatedOut) = buffers[1] {
            out = updatedOut
        }
        cur = out
        curN = nextN
        firstPass = false
    }
    B = cur
}
public func softmax_simd(
    A: [Float],
    B: inout [Float],
    global_max: [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == ((Int(n)+127)/128)*Int(b), "B has wrong size")
    precondition(global_max.count == Int(b), "global_max has wrong size")
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    var firstPass = true
    while curN > batch {
        let nextN = (curN + 127) / 128
        var out = [Float](repeating: 0, count: nextN * batch)
        var buffers: [KRBuffer]
        if firstPass {
            buffers = [
                .floatArray(cur),
                .floatArray(out),
                .floatArray(global_max),
                .uint32Val(UInt32(curN)),
                .uint32Val(b)
            ]
            kernel_runner_call(
                 "softmax_simd_reduce",
                buffers: &buffers,
                gridX: nextN, gridY: batch, gridZ: 1,
                tgX: 128, tgY: 1, tgZ: 1
            )
        } else {
            buffers = [
                .floatArray(cur),
                .floatArray(out),
                .uint32Val(UInt32(curN)),
                .uint32Val(b)
            ]
            kernel_runner_call(
                 "sum_simd_reduce",
                buffers: &buffers,
                gridX: nextN, gridY: batch, gridZ: 1,
                tgX: 128, tgY: 1, tgZ: 1
            )
        }
        if case .floatArray(let updatedOut) = buffers[1] {
            out = updatedOut
        }
        cur = out
        curN = nextN
        firstPass = false
    }
    B = cur
}
public func variance_simd(
    A: [Float],
    B: inout [Float],
    mu: [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == ((Int(n)+127)/128)*Int(b), "B has wrong size")
    precondition(mu.count == Int(b), "mu has wrong size")
    let batch = Int(b)
    var cur = A
    var curN = Int(n)
    var firstPass = true
    while curN > batch {
        let nextN = (curN + 127) / 128
        var out = [Float](repeating: 0, count: nextN * batch)
        var buffers: [KRBuffer]
        if firstPass {
            buffers = [
                .floatArray(cur),
                .floatArray(out),
                .floatArray(mu),
                .uint32Val(UInt32(curN)),
                .uint32Val(b)
            ]
            kernel_runner_call(
                 "variance_simd_reduce",
                buffers: &buffers,
                gridX: nextN, gridY: batch, gridZ: 1,
                tgX: 128, tgY: 1, tgZ: 1
            )
        } else {
            buffers = [
                .floatArray(cur),
                .floatArray(out),
                .uint32Val(UInt32(curN)),
                .uint32Val(b)
            ]
            kernel_runner_call(
                 "sum_simd_reduce",
                buffers: &buffers,
                gridX: nextN, gridY: batch, gridZ: 1,
                tgX: 128, tgY: 1, tgZ: 1
            )
        }
        if case .floatArray(let updatedOut) = buffers[1] {
            out = updatedOut
        }
        cur = out
        curN = nextN
        firstPass = false
    }
    B = cur
}
public func tanh(
    A: [Float],
    B: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "tanh",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedB) = buffers[1] { B = updatedB }
}
public func softlog(
    A: [Float],
    B: inout [Float],
    alpha: Float,
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .uint32Val(n),
        .floatVal(alpha),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "softlogx",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedB) = buffers[1] { B = updatedB }
}
public func relu(
    A: [Float],
    B: inout [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "relu",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedB) = buffers[1] { B = updatedB }
}
public func softmax(
    A: [Float],
    B: inout [Float],
    global_max: [Float],
    denom: [Float],
    n: UInt32,
    b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n*b), "B has wrong size")
    precondition(global_max.count == Int(b), "global_max has wrong size")
    precondition(denom.count == Int(b), "denom has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(global_max),
        .floatArray(denom),
        .uint32Val(n),
        .uint32Val(b)
    ]
    kernel_runner_call(
         "softmax",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedB) = buffers[1] { B = updatedB }
}
public func outer_prod(
    A: [Float],
    B: [Float],
    C: inout [Float],
    n: UInt32,
    m: UInt32,
    b: UInt32
){
    gemm1(
        A: A, B: B, C: &C,
        m: n, n: 1, p: m,
        b: b
    )
}

public func cortex_step(
    H_t0: [Float],
    W: [Float],
    H_t1: inout [Float],
    H_t1_out: inout [Float],
    k: UInt32,
    n: UInt32
){
    gemm1(H_t0, W, H_t1, k, n, n, 1)
    tanh(H_t1, H_t1, n*k, 1)
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
PLACEHOLDER */
kernel_runner_init()