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
public func add4(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ D: GPUBuffer <Float>,
    _ E: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    precondition(D.count == Int(n_*b_), "D has wrong size")
    precondition(E.count == Int(n_*b_), "E has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "add4",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .buffer(D.buffer),
            .buffer(E.buffer),
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
            width: 32,
            height: 32,
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
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            out=toggle?scratch0:scratch1
            toggle.toggle()
        }
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
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            out=toggle?scratch0:scratch1
            toggle.toggle()
        }
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
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            out=toggle?scratch0:scratch1
            toggle.toggle()
        }
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
    _ scratch0: GPUBuffer<Float>,
    _ scratch1: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(b_), "B has wrong size")
    let batch = Int(b_)
    var cur = A
    var curN = Int(n_)
    var toggle=false
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            out=toggle?scratch0:scratch1
            toggle.toggle()
        }
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
public func conv_r3(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 7, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "conv_r3",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 127) / 128,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
}
public func conv_r5(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 11, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "conv_r5",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 127) / 128,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
}
public func conv_r7(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 15, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "conv_r7",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 127) / 128,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
}
public func conv_r11(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 23, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "conv_r11",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 127) / 128,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
}
public func inhib_sub_r3(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ scratch0: GPUBuffer <Float>,
    _ scratch1: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 7, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "inhib_sub_r3",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 127) / 128,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
    sum_simd(stream: stream, B, scratch0, scratch1, C, n_, b_);
}
public func inhib_div_r7(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ W: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ scratch0: GPUBuffer <Float>,
    _ scratch1: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(W.count == 15, "W has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "inhib_div_r7",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&b, MemoryLayout<UInt32>.size)
        ],
        grid: MTLSize(
            width: (Int(n) + 127) / 128,
            height: Int(b),
            depth: 1
        ),
        threads: MTLSize(
            width: 128,
            height: 1,
            depth: 1
        )
    )
    sum_simd(stream: stream, B, scratch0, scratch1, C, n_, b_);
}
public func cortex_step(
    stream: ComputeStream,
    _ E_t: GPUBuffer <Float>,
    _ H_t0: GPUBuffer <Float>,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ X_g0: GPUBuffer <Float>,
    _ X_g: GPUBuffer <Float>,
    _ X_m3: GPUBuffer <Float>,
    _ X_m5: GPUBuffer <Float>,
    _ X_m7: GPUBuffer <Float>,
    _ X_m11: GPUBuffer <Float>,
    _ X_m: GPUBuffer <Float>,
    _ mu0: GPUBuffer <Float>,
    _ mu: GPUBuffer <Float>,
    _ gamma0: GPUBuffer <Float>,
    _ gamma: GPUBuffer <Float>,
    _ H_t1: GPUBuffer <Float>,
    _ W_conv_r3: GPUBuffer <Float>,
    _ W_conv_r5: GPUBuffer <Float>,
    _ W_conv_r7: GPUBuffer <Float>,
    _ W_conv_r11: GPUBuffer <Float>,
    _ W_inhib_sub_r3: GPUBuffer <Float>,
    _ W_inhib_div_r7: GPUBuffer <Float>,
    _ beta: GPUBuffer <Float>,
    _ softlog_alpha: Float,
    _ inhib_alpha: Float,
    _ n_: UInt32,
    _ k_: UInt32,
    _ r_: UInt32
){
    gemm(stream: stream, B, H_t0, X_g0, r_, n_, k_, 1);
    gemm(stream: stream, A, X_g0, X_g, n_, r_, k_, 1);
    conv_r3(stream: stream, H_t0, W_conv_r3, X_m3, n_, k_);
    conv_r5(stream: stream, H_t0, W_conv_r5, X_m5, n_, k_);
    conv_r7(stream: stream, H_t0, W_conv_r7, X_m7, n_, k_);
    conv_r11(stream: stream, H_t0, W_conv_r11, X_m11, n_, k_);
    add4(stream: stream, X_m3, X_m5, X_m7, X_m11, X_m, n_*k_, 1);
    inhib_sub_r3(stream: stream, H_t0, W_inhib_sub_r3, mu0, mu, n_, k_);
    inhib_div_r7(stream: stream, H_t0, W_inhib_div_r7, gamma0, gamma, n_, k_);
    var n=n_
    var k=k_
    stream.dispatch(
        kernel: "cortex_step",
        args: [
            .buffer(E_t.buffer),
            .buffer(H_t1.buffer),
            .buffer(X_g.buffer),
            .buffer(X_m.buffer),
            .buffer(mu.buffer),
            .buffer(gamma.buffer),
            .buffer(beta.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&k, MemoryLayout<UInt32>.size),
            .bytes(&softlog_alpha, MemoryLayout<Float>.size),
            .bytes(&inhib_alpha, MemoryLayout<Float>.size)
        ],
        grid: MTLSize(
            width: (Int(k)+255)/256,
            height: Int(n),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
    stream.advance()
}
public func final_oja_step(
    stream: ComputeStream,
    _ G1: GPUBuffer <Float>,
    _ G2: GPUBuffer <Float>,
    _ eta: GPUBuffer <Float>,
    _ A: GPUBuffer <Float>,
    _ A_new: GPUBuffer <Float>,
    _ n_: UInt32,
    _ r_: UInt32,
    _ lambda_: Float
){
    precondition(G1.count == Int(n_*r_), "G1 has wrong size")
    precondition(G2.count == Int(n_*r_), "G2 has wrong size")
    precondition(eta.count == Int(n_), "eta has wrong size")
    precondition(A.count == Int(n_*r_), "A has wrong size")
    precondition(A_new.count == Int(n_*r_), "A_new has wrong size")
    var n=n_
    var r=r_
    var lambda=lambda_
    stream.dispatch(
        kernel: "final_oja_step",
        args: [
            .buffer(G1.buffer),
            .buffer(G2.buffer),
            .buffer(eta.buffer),
            .buffer(A.buffer),
            .buffer(A_new.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&r, MemoryLayout<UInt32>.size),
            .bytes(&lambda, MemoryLayout<Float>.size)
        ],
        grid: MTLSize(
            width: (Int(r) + 255) / 256,
            height: Int(n),
            depth: 1
        ),
        threads: MTLSize(
            width: 256,
            height: 1,
            depth: 1
        )
    )
}
public func get_eta(
    stream: ComputeStream,
    _ NE: GPUBuffer <Float>,
    _ ACh: GPUBuffer <Float>,
    _ DA: GPUBuffer <Float>,
    _ eta: GPUBuffer <Float>,
    _ n_: UInt32,
    _ w_NE_: Float,
    _ w_ACh_: Float,
    _ w_DA_: Float,
    _ b_: Float,
    _ eta_max_: Float
){
    precondition(NE.count == Int(n_), "NE has wrong size")
    precondition(ACh.count == Int(n_), "ACh has wrong size")
    precondition(DA.count == Int(n_), "DA has wrong size")
    precondition(eta.count == Int(n_), "eta has wrong size")
    var n=n_
    var w_NE=w_NE_
    var w_ACh=w_ACh_
    var w_DA=w_DA_
    var b=b_
    var eta_max=eta_max_
    stream.dispatch(
        kernel: "get_eta",
        args: [
            .buffer(NE.buffer),
            .buffer(ACh.buffer),
            .buffer(DA.buffer),
            .buffer(eta.buffer),
            .bytes(&n, MemoryLayout<UInt32>.size),
            .bytes(&w_NE, MemoryLayout<Float>.size),
            .bytes(&w_ACh, MemoryLayout<Float>.size),
            .bytes(&w_DA, MemoryLayout<Float>.size),
            .bytes(&b, MemoryLayout<Float>.size),
            .bytes(&eta_max, MemoryLayout<Float>.size)
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
//NB: ASSUME B_t IS ALREADY SET (given that B is constant)
public func oja(
    stream: ComputeStream,
    _ A: GPUBuffer<Float>,
    _ A_new: GPUBuffer<Float>,
    _ B: GPUBuffer<Float>,
    _ eta: GPUBuffer<Float>,
    _ B_t: GPUBuffer<Float>,
    _ T1: GPUBuffer<Float>,
    _ T2: GPUBuffer<Float>,
    _ S: GPUBuffer<Float>,
    _ U: GPUBuffer<Float>,
    _ G: GPUBuffer<Float>,
    _ H: GPUBuffer<Float>,
    _ X: GPUBuffer<Float>,
    _ Y: GPUBuffer<Float>,
    _ Y_t: GPUBuffer<Float>,
    _ r_: UInt32,
    _ n_: UInt32,
    _ k_: UInt32,
    _ lambda_: Float,
){
    transpose(stream: stream, Y, Y_t, n_, k_)
    gemm(stream: stream, Y_t, B, T1, k_, n_, r_, 1)
    gemm(stream: stream, Y_t, A, T2, k_, n_, r_, 1)
    gemm(stream: stream, B_t, B, S, r_, n_, r_, 1)
    gemm(stream: stream, T2, S, U, k_, r_, r_, 1)
    gemm(stream: stream, X, T1, G, n_, k_, r_, 1)
    gemm(stream: stream, Y, U, H, n_, k_, r_, 1)
    final_oja_step(stream: stream, G, H, eta, A, A_new, n_, r_, lambda_)
    copy(stream: stream, A_new, A, n_*r_, 1)
}