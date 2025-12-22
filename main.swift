import Metal
import Foundation
import Darwin
//<start AI_WRITTEN>
// ============================================================
// 1. IMMUTABLE METAL CONTEXT (THREAD-SAFE, SHARED)
// ============================================================

public final class MetalContext {

    public let device: MTLDevice
    public let library: MTLLibrary
    public let pipelines: [String: MTLComputePipelineState]

    public init(metallibURL: URL, kernelNames: [String]) {
        self.device = MTLCreateSystemDefaultDevice()!
        self.library = try! device.makeLibrary(URL: metallibURL)

        var tmp: [String: MTLComputePipelineState] = [:]
        for name in kernelNames {
            let fn = library.makeFunction(name: name)!
            tmp[name] = try! device.makeComputePipelineState(function: fn)
        }
        self.pipelines = tmp
    }
}

// ============================================================
// 2. PERSISTENT GPU BUFFER (LOGICAL SIZE, FIXED CAPACITY)
// ============================================================

public final class GPUBuffer<T> {

    public let buffer: MTLBuffer
    public let capacity: Int
    public var count: Int

    public init(device: MTLDevice, capacity: Int) {
        self.capacity = capacity
        self.count = capacity
        self.buffer = device.makeBuffer(
            length: capacity * MemoryLayout<T>.stride,
            options: .storageModeShared
        )!
    }

    @inline(__always)
    public func ptr() -> UnsafeMutablePointer<T> {
        buffer.contents().assumingMemoryBound(to: T.self)
    }
}

// ============================================================
// 3. ZERO-ALLOCATION KERNEL ARGUMENTS
// ============================================================

public enum KernelArg {
    case buffer(MTLBuffer)
    case bytes(UnsafeRawPointer, Int)
}

// ============================================================
// 4. COMPUTE STREAM (ONE PER COGNITIVE PROCESS)
// ============================================================

public final class ComputeStream {

    private let ctx: MetalContext
    private let queue: MTLCommandQueue

    private let inflight: Int
    private var cmdRing: [MTLCommandBuffer] = []
    private var encRing: [MTLComputeCommandEncoder] = []

    private var index: Int = 0

    // --------------------------------------------------------

    public init(context: MetalContext, inflightBuffers: Int = 3) {
        self.ctx = context
        self.queue = context.device.makeCommandQueue()!
        self.inflight = inflightBuffers

        for _ in 0..<inflight {
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            cmdRing.append(cmd)
            encRing.append(enc)
        }
    }

    // --------------------------------------------------------
    // HOT PATH — NO ALLOCATION, NO SYNC
    // --------------------------------------------------------

    @inline(__always)
    public func dispatch(
        kernel: String,
        args: [KernelArg],
        grid: MTLSize,
        threads: MTLSize
    ) {
        let enc = encRing[index]
        let pipe = ctx.pipelines[kernel]!

        enc.setComputePipelineState(pipe)

        for (i, arg) in args.enumerated() {
            switch arg {
            case .buffer(let b):
                enc.setBuffer(b, offset: 0, index: i)
            case .bytes(let ptr, let size):
                enc.setBytes(ptr, length: size, index: i)
            }
        }

        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
    }

    // --------------------------------------------------------
    // SUBMIT CURRENT STEP (NO WAIT)
    // --------------------------------------------------------

    @inline(__always)
    public func advance() {
        let cmd = cmdRing[index]
        let enc = encRing[index]

        enc.endEncoding()
        cmd.commit()

        index = (index + 1) % inflight

        let nextCmd = queue.makeCommandBuffer()!
        let nextEnc = nextCmd.makeComputeCommandEncoder()!

        cmdRing[index] = nextCmd
        encRing[index] = nextEnc
    }

    // --------------------------------------------------------
    // EXPLICIT SYNCHRONIZATION (BOUNDARY ONLY)
    // --------------------------------------------------------

    public func synchronize() {
        for cmd in cmdRing {
            cmd.waitUntilCompleted()
        }
    }
}
//<end AI_WRITTEN>
public func add(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "add",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func copy(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "copy",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func transpose(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ m_: UInt32
){
    precondition(A.count == Int(n_*m_), "A has wrong size")
    precondition(B.count == Int(n_*m_), "B has wrong size")
    var n=n_
    var m=m_
    stream.dispatch(
        kernel: "transpose",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&m, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 31) / 32,
            height: (Int(m) + 31) / 32,
            depth: 1
        ),
        threads: MTLSize(
            width: 64,
            height: 64,
            depth: 1
        )
    )
}
public func div(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "div",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func mul(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "mul",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func sub(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "sub",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}

public func embedding(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <UInt32>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ d_: UInt32,
    _ vocab_size_: UInt32
){
    precondition(A.count == Int(vocab_size_*d_), "A has wrong size")
    precondition(B.count == Int(n_), "B has wrong size")
    precondition(C.count == Int(n_*d_), "C has wrong size")
    var n=n_
    var d=d_
    var vocab_size=vocab_size_
    stream.dispatch(
        kernel: "embedding",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&d, MemoryLayout<UInt32>.size),
            .bytes(&vocab_size, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: 1,
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func gemm(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ m_: UInt32,
    _ n_: UInt32,
    _ p_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(m_*n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*p_*b_), "B has wrong size")
    precondition(C.count == Int(m_*p_*b_), "C has wrong size")
    var m=m_
    var n=n_
    var p=p_
    var b=b_
    stream.dispatch(
        kernel: "gemm",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .bytes(&m, MemoryLayout<UInt32>.size),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&p, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: ((Int(p)+63)/64),
            height: ((Int(m)+63)/64)*Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 8,
            height: 64,
            depth: 1
        )
    )
}
public func layernorm(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ mu: GPUBuffer <Float>,
    _ sigma2: GPUBuffer <Float>,
    _ n_: UInt32,
    _ eps_: Float,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(mu.count == Int(b_), "mu has wrong size")
    precondition(sigma2.count == Int(b_), "sigma2 has wrong size")
    var n=n_
    var eps=eps_
    var b=b_
    stream.dispatch(
        kernel: "layernorm",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(mu.buffer),
            .buffer(sigma2.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&eps, MemoryLayout<Float>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func tanh(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "tanh",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func softlog(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ alpha_: Float,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var alpha=alpha_
    var b=b_
    stream.dispatch(
        kernel: "softlog",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&alpha, MemoryLayout<Float>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func relu(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "relu",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func softmax(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ global_max: GPUBuffer<Float>,
    _ denom: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(global_max.count == Int(b_), "global_max has wrong size")
    precondition(denom.count == Int(b_), "denom has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "softmax",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(global_max.buffer),
            .buffer(denom.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 255) / 256,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func outer_prod(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ C: GPUBuffer<Float>,
    _ n: UInt32,
    _ m: UInt32,
    _ b: UInt32
){
    gemm(
        stream: stream,
        A, B, C,
        n, 1, m,
        b
    )
}
public func max_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out = nextN == 1 ? B : GPUBuffer<Float>(device: stream.device, capacity: nextN * batch)
        var n = UInt32(curN)
        var b = b_
        stream.dispatch(
            kernel: "max_simd_reduce",
            args: [
                .buffer(cur.buffer),
                .buffer(out.buffer),
                .bytes(&n, MemoryLayout<UInt32>.size),
                .bytes(&b, MemoryLayout<UInt32>.size)
            ],
            grid: MTLSize(
                width: nextN,
                height: batch,
                depth: 1
            ),
            threads: MTLSize(
                width: 128,
                height: 1,
                depth: 1
            )
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func sum_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out = nextN == 1 ? B : GPUBuffer<Float>(device: stream.device, capacity: nextN * batch)
        var n = UInt32(curN)
        var b = b_
        stream.dispatch(
            kernel: "sum_simd_reduce",
            args: [
                .buffer(cur.buffer),
                .buffer(out.buffer),
                .bytes(&n, MemoryLayout<UInt32>.size),
                .bytes(&b, MemoryLayout<UInt32>.size)
            ],
            grid: MTLSize(
                width: nextN,
                height: batch,
                depth: 1
            ),
            threads: MTLSize(
                width: 128,
                height: 1,
                depth: 1
            )
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func mean_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out = nextN == 1 ? B : GPUBuffer<Float>(device: stream.device, capacity: nextN * batch)
        var n = UInt32(curN)
        var b = b_
        stream.dispatch(
            kernel: "mean_simd_reduce",
            args: [
                .buffer(cur.buffer),
                .buffer(out.buffer),
                .bytes(&n, MemoryLayout<UInt32>.size),
                .bytes(&b, MemoryLayout<UInt32>.size)
            ],
            grid: MTLSize(
                width: nextN,
                height: batch,
                depth: 1
            ),
            threads: MTLSize(
                width: 128,
                height: 1,
                depth: 1
            )
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
public func abs_mean_simd(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out = nextN == 1 ? B : GPUBuffer<Float>(device: stream.device, capacity: nextN * batch)
        var n = UInt32(curN)
        var b = b_
        stream.dispatch(
            kernel: "abs_mean_simd_reduce",
            args: [
                .buffer(cur.buffer),
                .buffer(out.buffer),
                .bytes(&n, MemoryLayout<UInt32>.size),
                .bytes(&b, MemoryLayout<UInt32>.size)
            ],
            grid: MTLSize(
                width: nextN,
                height: batch,
                depth: 1
            ),
            threads: MTLSize(
                width: 128,
                height: 1,
                depth: 1
            )
        )
        if nextN == 1 { return }
        cur = out
        curN = nextN
    }
}
