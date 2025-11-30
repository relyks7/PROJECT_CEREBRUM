import Metal
import Foundation

class KernelRunner {
    let device: MTLDevice
    let queue: MTLCommandQueue
    var pipelines: [String: MTLComputePipelineState] = [:]
    let kernelDir: String

    init(device: MTLDevice, kernelDirectory: String = "kernels") {
        self.device = device
        self.queue = device.makeCommandQueue()!
        self.kernelDir = kernelDirectory
    }

    func pipeline(for name: String) -> MTLComputePipelineState {
        if let p = pipelines[name] {
            return p
        }

        let path = "\(kernelDir)/\(name).metallib"
        let url = URL(fileURLWithPath: path)

        let lib = try! device.makeLibrary(URL: url)
        let fn  = lib.makeFunction(name: name)!
        let pso = try! device.makeComputePipelineState(function: fn)

        pipelines[name] = pso
        return pso
    }

    func run(
        _ name: String,
        buffers: [MTLBuffer],
        grid: MTLSize,
        threads: MTLSize
    ) {
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!

        let pso = pipeline(for: name)
        enc.setComputePipelineState(pso)

        for (i, buf) in buffers.enumerated() {
            enc.setBuffer(buf, offset: 0, index: i)
        }

        enc.dispatchThreads(grid, threadsPerThreadgroup: threads)
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()
    }
}