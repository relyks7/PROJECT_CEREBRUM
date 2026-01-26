import Metal
import Foundation
import Darwin

//<start AI_WRITTEN>
// ============================================================
// 0. SMALL HELPERS
// ============================================================

@inline(__always)
func bytes<T>(_ value: inout T) -> KernelArg {
    return withUnsafeBytes(of: &value) {
        KernelArg.bytes($0.baseAddress!, $0.count)
    }
}

@inline(__always)
func load(_ src: [Float], into dst: GPUBuffer<Float>) {
    precondition(src.count <= dst.capacity)
    dst.ptr().update(from: src, count: src.count)
}

// ============================================================
// 1. METAL CONTEXT (PIPELINES RESOLVED ONCE)
// ============================================================

public final class MetalContext {

    public let device: MTLDevice
    public let libraries: [MTLLibrary]

    private let pipelines: [String: MTLComputePipelineState]

    @inline(__always)
    public func pipeline(_ name: String) -> MTLComputePipelineState {
        pipelines[name]!
    }

    public init(kernelsDirectory: URL) {
        self.device = MTLCreateSystemDefaultDevice()!

        let fm = FileManager.default
        let urls = (try? fm.contentsOfDirectory(
            at: kernelsDirectory,
            includingPropertiesForKeys: nil
        ))?.filter { $0.pathExtension == "metallib" } ?? []

        precondition(!urls.isEmpty, "No .metallib files found")

        var libs: [MTLLibrary] = []
        for url in urls {
            libs.append(try! device.makeLibrary(URL: url))
        }
        self.libraries = libs

        var pipeTable: [String: MTLComputePipelineState] = [:]
        for lib in libs {
            for name in lib.functionNames where pipeTable[name] == nil {
                let fn = lib.makeFunction(name: name)!
                pipeTable[name] = try! device.makeComputePipelineState(function: fn)
            }
        }

        precondition(!pipeTable.isEmpty, "No compute kernels discovered")
        self.pipelines = pipeTable
    }
}

// ============================================================
// 2. GPU BUFFER (FIXED CAPACITY, SHARED MEMORY)
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
// 3. KERNEL ARG ENUM (NO ALLOCATIONS)
// ============================================================

public enum KernelArg {
    case buffer(MTLBuffer)
    case bytes(UnsafeRawPointer, Int)
}

// ============================================================
// 4. COMPUTE STREAM (PURE PSO DISPATCH)
// ============================================================

public final class ComputeStream {

    private let queue: MTLCommandQueue
    private let inflight: Int

    private var cmdRing: [MTLCommandBuffer]
    private var encRing: [MTLComputeCommandEncoder]
    private var committed: [Bool]

    private var index: Int = 0

    // --------------------------------------------------------

    public init(context: MetalContext, inflightBuffers: Int = 3) {
        self.queue = context.device.makeCommandQueue()!
        self.inflight = inflightBuffers

        self.cmdRing = []
        self.encRing = []
        self.committed = Array(repeating: false, count: inflight)

        for _ in 0..<inflight {
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            cmdRing.append(cmd)
            encRing.append(enc)
        }
    }

    // --------------------------------------------------------
    // 🔥 HOT PATH — DIRECT PIPELINE STATE
    // --------------------------------------------------------

    @inline(__always)
    public func dispatch(
        pipeline: MTLComputePipelineState,
        args: [KernelArg],
        grid: MTLSize,
        threads: MTLSize
    ) {
        let enc = encRing[index]
        enc.setComputePipelineState(pipeline)

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
    // SUBMIT + ROTATE RING
    // --------------------------------------------------------

    @inline(__always)
    public func advance() {
        let cmd = cmdRing[index]
        let enc = encRing[index]

        enc.endEncoding()
        cmd.commit()
        committed[index] = true

        index = (index + 1) % inflight

        if committed[index] {
            cmdRing[index].waitUntilCompleted()
            committed[index] = false
        }

        let nextCmd = queue.makeCommandBuffer()!
        let nextEnc = nextCmd.makeComputeCommandEncoder()!

        cmdRing[index] = nextCmd
        encRing[index] = nextEnc
    }

    // --------------------------------------------------------
    // FULL BARRIER (BOUNDARY ONLY)
    // --------------------------------------------------------

    public func synchronize() {
        for i in 0..<inflight where committed[i] {
            cmdRing[i].waitUntilCompleted()
            committed[i] = false
        }
    }
}
//<end AI_WRITTEN>