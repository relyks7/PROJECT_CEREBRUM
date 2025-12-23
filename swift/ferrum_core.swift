import Metal
import Foundation
import Darwin
//<start AI_WRITTEN>
// ============================================================
// 1. IMMUTABLE METAL CONTEXT (MULTI-METALLIB, THREAD-SAFE)
// ============================================================
@inline(__always)
func bytes<T>(_ value: inout T) -> KernelArg {
    return withUnsafeBytes(of: &value) {
        KernelArg.bytes($0.baseAddress!, $0.count)
    }
}
func load(_ src: [Float], into dst: GPUBuffer<Float>) {
    precondition(src.count <= dst.capacity)
    dst.ptr().update(from: src, count: src.count)
}
public final class MetalContext {

    public let device: MTLDevice
    public let libraries: [MTLLibrary]
    public let pipelines: [String: MTLComputePipelineState]

    /// Automatically loads all `.metallib` files from a directory
    public init(kernelsDirectory: URL) {
        self.device = MTLCreateSystemDefaultDevice()!

        let fm = FileManager.default

        // 1. Discover metallibs
        let urls = (try? fm.contentsOfDirectory(
            at: kernelsDirectory,
            includingPropertiesForKeys: nil
        ))?.filter { $0.pathExtension == "metallib" } ?? []

        precondition(!urls.isEmpty, "No .metallib files found in \(kernelsDirectory.path)")

        // 2. Load libraries
        var libs: [MTLLibrary] = []
        for url in urls {
            let lib = try! device.makeLibrary(URL: url)
            libs.append(lib)
        }
        self.libraries = libs

        // 3. Discover kernels + build pipelines
        var pipeTable: [String: MTLComputePipelineState] = [:]

        for lib in libs {
            for name in lib.functionNames {
                // Skip duplicates (first wins)
                if pipeTable[name] != nil { continue }

                guard let fn = lib.makeFunction(name: name) else { continue }

                let pipe = try! device.makeComputePipelineState(function: fn)
                pipeTable[name] = pipe
            }
        }

        precondition(!pipeTable.isEmpty, "No compute kernels found in metallibs")

        self.pipelines = pipeTable
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
    private var committed: [Bool] = []

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
            committed.append(false)
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
    // SUBMIT CURRENT STEP (SAFE RING ADVANCE)
    // --------------------------------------------------------

    @inline(__always)
    public func advance() {
        // Commit current slot
        let cmd = cmdRing[index]
        let enc = encRing[index]

        enc.endEncoding()
        cmd.commit()
        committed[index] = true

        // Advance ring
        index = (index + 1) % inflight

        // Ensure slot is free before reuse
        if committed[index] {
            cmdRing[index].waitUntilCompleted()
            committed[index] = false
        }

        // Create new command buffer & encoder
        let nextCmd = queue.makeCommandBuffer()!
        let nextEnc = nextCmd.makeComputeCommandEncoder()!

        cmdRing[index] = nextCmd
        encRing[index] = nextEnc
    }

    // --------------------------------------------------------
    // EXPLICIT SYNCHRONIZATION (BOUNDARY ONLY)
    // --------------------------------------------------------

    public func synchronize() {
        for i in 0..<inflight {
            if committed[i] {
                cmdRing[i].waitUntilCompleted()
                committed[i] = false
            }
        }
    }
}
//<end AI_WRITTEN>