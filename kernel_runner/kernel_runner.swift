import Foundation
import Metal

var kernel_runner_device: MTLDevice!
var kernel_runner_obj: KernelRunner!

@_cdecl("kernel_runner_init")
public func kernel_runner_init() {
    kernel_runner_device = MTLCreateSystemDefaultDevice()
    kernel_runner_obj = KernelRunner(device: kernel_runner_device)
    print("kernel_runner initialized on \(kernel_runner_device.name)")
}


/// UNIVERSAL GPU CALL ENTRY
/// Accepts ANY number of buffers (ptrArray), ANY kernel name, ANY grid/threadgroup sizes.
@_cdecl("kernel_runner_call")
public func kernel_runner_call(
    kernelNamePtr: UnsafePointer<CChar>,
    ptrArray: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>?,
    lenArray: UnsafeMutablePointer<Int32>?,
    numBuffers: Int32,
    gridX: Int32, gridY: Int32, gridZ: Int32,
    tgX: Int32, tgY: Int32, tgZ: Int32
) {
    let kernelName = String(cString: kernelNamePtr)

    var buffers: [MTLBuffer] = []

    // turn ptrs → Metal buffers
    for i in 0..<numBuffers {
        let ptr  = ptrArray![Int(i)]
        let len  = lenArray![Int(i)]

        if let p = ptr {
            let buf = kernel_runner_device.makeBuffer(
                bytes: p,
                length: Int(len) * MemoryLayout<Float>.size
            )!
            buffers.append(buf)
        }
    }

    let grid = MTLSize(
        width:  Int(gridX),
        height: Int(gridY),
        depth:  Int(gridZ)
    )

    let threads = MTLSize(
        width:  Int(tgX),
        height: Int(tgY),
        depth:  Int(tgZ)
    )

    // run the Metal kernel
    kernel_runner_obj.run(kernelName, buffers: buffers, grid: grid, threads: threads)

    // copy back into caller memory
    for i in 0..<numBuffers {
        let ptr = ptrArray![Int(i)]
        let len = lenArray![Int(i)]
        if let p = ptr {
            memcpy(p, buffers[Int(i)].contents(), Int(len) * MemoryLayout<Float>.size)
        }
    }
}