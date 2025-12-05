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
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
public func inhib_sub(
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ alpha: Float,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
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
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func inhib_div(
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ alpha: Float,
    _ eps: Float,
    _ n: UInt32,
    _ b: UInt32
){
    precondition(A.count == Int(n*b), "A has wrong size")
    precondition(B.count == Int(n), "B has wrong size")
    precondition(C.count == Int(n*b), "C has wrong size")
    var buffers: [KRBuffer]=[
        .floatArray(A),
        .floatArray(B),
        .floatArray(C),
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
    if case .floatArray(let updatedC) = buffers[2] { C = updatedC }
}
public func embedding(
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ n: UInt32,
    _ d: UInt32,
    _ vocab_size: UInt32
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
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ m: UInt32,
    _ n: UInt32,
    _ p: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ m: UInt32,
    _ n: UInt32,
    _ p: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ m: UInt32,
    _ n: UInt32,
    _ p: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: inout [Float],
    _ mu: [Float],
    _ sigma2: [Float],
    _ n: UInt32,
    _ eps: Float,
    _ b: UInt32
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
    _ A: [Float],
    _ B: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
public func abs_mean_simd(
    _ A: [Float],
    _ B: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
        let kernelName = firstPass ? "abs_mean_simd_reduce" : "sum_simd_reduce"
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
    _ A: [Float],
    _ B: inout [Float],
    _ global_max: [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: inout [Float],
    _ mu: [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: inout [Float],
    _ alpha: Float,
    _ n: UInt32,
    _ b: UInt32
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
         "softlog",
        buffers: &buffers,
        gridX: (Int(n)+255)/256, gridY: Int(b), gridZ:1,
        tgX: 256, tgY: 1, tgZ: 1
    )
    if case .floatArray(let updatedB) = buffers[1] { B = updatedB }
}
public func relu(
    _ A: [Float],
    _ B: inout [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: inout [Float],
    _ global_max: [Float],
    _ denom: [Float],
    _ n: UInt32,
    _ b: UInt32
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
    _ A: [Float],
    _ B: [Float],
    _ C: inout [Float],
    _ n: UInt32,
    _ m: UInt32,
    _ b: UInt32
){
    gemm1(
        A, B, &C,
        n, 1, m,
        b
    )
}
public func cortex_step(
    H_t0: [Float], 
    A: [Float],
    B: [Float],
    H_t1: inout [Float],
    alpha_sub: Float,
    alpha_div: Float,
    k: UInt32,
    r: UInt32,
    n: UInt32
){
    var H_raw_inter = Array(repeating: 0.0, count: Int(k*r))
    gemm1(H_t0, A, &H_raw_inter, k, n, r, 1)
    var H_raw_ipt=Array(repeating:0.0, count:Int(k*n))
    gemm2(H_raw_inter, B, &H_raw_ipt, k, r, n, 1)
    var H_raw=Array(repeating:0.0, count:Int(k*n))
    softlog(H_raw_ipt, &H_raw, 1.7, k*n, 1)
    var mu=Array(repeating:0.0, count:Int(n))
    mean_simd(H_raw, &mu, k, n)
    var gamma=Array(repeating:0.0, count:Int(n))
    abs_mean_simd(H_raw, &gamma, k, n)
    var H_sub=Array(repeating:0.0, count:Int(n*k))
    inhib_sub(H_raw, mu, &H_sub, alpha_sub, n, k)
    inhib_div(H_sub, gamma, &H_t1, alpha_div, eps, n, k)
}
public func fast_oja(
    A: inout [Float],
    B: inout [Float],
    U: [Float],
    V: [Float],
    X: [Float],
    Y: [Float],
    etanull: Float,
    k: UInt32,
    n: UInt32,
    r: UInt32
){
    // ----------------------------------------------------
    // 1. softlog(U), softlog(V)
    // ----------------------------------------------------
    var U_t = [Float](repeating: 0, count: Int(n*r))
    var V_t = [Float](repeating: 0, count: Int(n*r))
    softlog(U, &U_t, 1.7, n, r)
    softlog(V, &V_t, 1.7, n, r)

    // M = etanull * (U_t ⊙ V_t)
    var M = [Float](repeating: 0, count: Int(n*r))
    mul(U_t, V_t, &M, n, r)

    var eta_arr = [Float](repeating: etanull, count: Int(n*r))
    mul(M, eta_arr, &M, n, r)


    // ----------------------------------------------------
    // 2. Compute p = X B    (k×n) × (n×r) = k×r
    // ----------------------------------------------------
    var p = [Float](repeating: 0, count: Int(k*r))
    gemm1(X, B, &p, k, n, r, 1)


    // ----------------------------------------------------
    // 3. Compute q = Y A    (k×n) × (n×r) = k×r
    // ----------------------------------------------------
    var q = [Float](repeating: 0, count: Int(k*r))
    gemm1(Y, A, &q, k, n, r, 1)


    // ----------------------------------------------------
    // 4. Compute Y^T p     (n×k) × (k×r) = n×r
    //    This is the "Hebb" term
    // ----------------------------------------------------
    var YTp = [Float](repeating: 0, count: Int(n*r))
    gemm3(Y, p, &YTp, n, k, r, 1)
    // gemm3 is A^T B, so Y^T p


    // ----------------------------------------------------
    // 5. Compute Y^T q     (n×k) × (k×r) = n×r
    //    This is the Oja normalization term
    // ----------------------------------------------------
    var YTq = [Float](repeating: 0, count: Int(n*r))
    gemm3(Y, q, &YTq, n, k, r, 1)


    // ----------------------------------------------------
    // 6. G = Y^T p - Y^T q   (n×r)
    // ----------------------------------------------------
    var G = [Float](repeating: 0, count: Int(n*r))
    sub(YTp, YTq, &G, n, r)


    // ----------------------------------------------------
    // 7. ΔA = M ⊙ G
    // ----------------------------------------------------
    var dA = [Float](repeating: 0, count: Int(n*r))
    mul(M, G, &dA, n, r)


    // ----------------------------------------------------
    // 8. ΔB = M ⊙ G   (symmetric update)
    // ----------------------------------------------------
    var dB = [Float](repeating: 0, count: Int(n*r))
    mul(M, G, &dB, n, r)


    // ----------------------------------------------------
    // 9. Apply updates
    // ----------------------------------------------------
    add(A, dA, &A, n, r)
    add(B, dB, &B, n, r)
}
public func slow_oja(
    U: inout [Float],
    V: inout [Float],
    A: [Float],
    B: [Float],
    X: [Float],
    Y: [Float],
    E: [Float],
    eta0: Float,
    lambda: Float,
    beta: Float,
    n: UInt32,
    r: UInt32,
    k: UInt32
) {
    // 0) Compute global surprise δ from E (on CPU)
    let kn = Int(k * n)
    var absSum: Float = 0
    for i in 0..<kn { absSum += abs(E[i]) }
    let deltaRaw = absSum / Float(kn)
    let delta = tanh(beta * deltaRaw)  // global modulatory factor

    // 1) softlog(U), softlog(V) → U_t, V_t
    var U_t = [Float](repeating: 0, count: Int(n*r))
    var V_t = [Float](repeating: 0, count: Int(n*r))
    softlog(U, &U_t, 1.7, n, r)
    softlog(V, &V_t, 1.7, n, r)

    // 2) M = eta0 * (U_t ⊙ V_t)
    var M = [Float](repeating: 0, count: Int(n*r))
    mul(U_t, V_t, &M, n, r)
    var etaArr = [Float](repeating: eta0, count: Int(n*r))
    mul(M, etaArr, &M, n, r)

    // 3) Compute G = Yᵀ(XB) − Yᵀ(YA)
    var p = [Float](repeating: 0, count: Int(k*r))
    gemm1(X, B, &p, k, n, r, 1)

    var q = [Float](repeating: 0, count: Int(k*r))
    gemm1(Y, A, &q, k, n, r, 1)

    var YTp = [Float](repeating: 0, count: Int(n*r))
    gemm3(Y, p, &YTp, n, k, r, 1)

    var YTq = [Float](repeating: 0, count: Int(n*r))
    gemm3(Y, q, &YTq, n, k, r, 1)

    var G = [Float](repeating: 0, count: Int(n*r))
    sub(YTp, YTq, &G, n, r)

    // 4) H_fast = M ⊙ G
    var H_fast = [Float](repeating: 0, count: Int(n*r))
    mul(M, G, &H_fast, n, r)

    // 5) S = δ * (H_fast ⊙ G)
    var S = [Float](repeating: 0, count: Int(n*r))
    mul(H_fast, G, &S, n, r)
    if delta != 0 {
        var deltaArr = [Float](repeating: delta, count: Int(n*r))
        mul(S, deltaArr, &S, n, r)
    }

    // 6) ΔU = λ * S V
    var dU = [Float](repeating: 0, count: Int(n*r))
    gemm1(S, V, &dU, n, r, r, 1)

    // 7) ΔV = λ * Sᵀ U
    var dV = [Float](repeating: 0, count: Int(n*r))
    gemm3(S, U, &dV, n, r, r, 1)

    var lamArr = [Float](repeating: lambda, count: Int(n*r))
    mul(dU, lamArr, &dU, n, r)
    mul(dV, lamArr, &dV, n, r)

    add(U, dU, &U, n, r)
    add(V, dV, &V, n, r)
}

kernel_runner_init()