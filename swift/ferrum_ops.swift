import Metal
import Foundation
import Darwin
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
            bytes(&n),
            bytes(&b)
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
    _ b_: UInt32,
    _ alpha_: Float
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    precondition(D.count == Int(n_*b_), "D has wrong size")
    precondition(E.count == Int(n_*b_), "E has wrong size")
    var n=n_
    var b=b_
    var alpha=alpha_
    stream.dispatch(
        kernel: "add4",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            .buffer(D.buffer),
            .buffer(E.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&alpha)
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
public func axbpy(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ B: GPUBuffer <Float>,
    _ C: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32,
    _ Y_: Float,
    _ l_: Float
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    precondition(B.count == Int(n_*b_), "B has wrong size")
    precondition(C.count == Int(n_*b_), "C has wrong size")
    var n=n_
    var b=b_
    var Y=Y_
    var l=l_
    stream.dispatch(
        kernel: "axbpy",
        args: [
            .buffer(A.buffer),
            .buffer(B.buffer),
            .buffer(C.buffer),
            bytes(&n),
            bytes(&b),
            bytes(&Y),
            bytes(&l)
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
            bytes(&n),
            bytes(&b)
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
public func zero(
    stream: ComputeStream,
    _ A: GPUBuffer <Float>,
    _ n_: UInt32,
    _ b_: UInt32
){
    precondition(A.count == Int(n_*b_), "A has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "zero",
        args: [
            .buffer(A.buffer),
            bytes(&n),
            bytes(&b)
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
            bytes(&n),
            bytes(&m)
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
            bytes(&n),
            bytes(&b)
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
            bytes(&n),
            bytes(&b)
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
            bytes(&n),
            bytes(&b)
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
            bytes(&n),
            bytes(&d),
            bytes(&vocab_size)
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
            bytes(&m),
            bytes(&n),
            bytes(&p),
            bytes(&b)
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
            bytes(&n),
            bytes(&eps),
            bytes(&b)
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
            bytes(&n),
            bytes(&b)
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
            bytes(&n),
            bytes(&alpha),
            bytes(&b)
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
            bytes(&n),
            bytes(&b)
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
            bytes(&n),
            bytes(&b)
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
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        stream.dispatch(
            kernel: "max_simd_reduce",
            args: [
                .buffer(cur.buffer),
                .buffer(out.buffer),
                bytes(&n),
                bytes(&b)
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
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        stream.dispatch(
            kernel: "sum_simd_reduce",
            args: [
                .buffer(cur.buffer),
                .buffer(out.buffer),
                bytes(&n),
                bytes(&b)
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
    var isfirst=true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        if isfirst{
            stream.dispatch(
                kernel: "mean_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
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
            isfirst=false
        }
        else{
            stream.dispatch(
                kernel: "sum_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
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
        }
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
    var isfirst=true
    while curN > 1 {
        let nextN = (curN + 127) / 128
        let out: GPUBuffer<Float>
        if nextN == 1{
            out=B
        } else{
            if toggle{
                out=scratch0
            }else{
                out=scratch1
            }
            toggle.toggle()
        }
        var n = UInt32(curN)
        var b = b_
        if isfirst{
            stream.dispatch(
                kernel: "abs_mean_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
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
            isfirst=false
        }
        else{
            stream.dispatch(
                kernel: "sum_simd_reduce",
                args: [
                    .buffer(cur.buffer),
                    .buffer(out.buffer),
                    bytes(&n),
                    bytes(&b)
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
        }
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
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
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
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
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
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
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
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
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
    precondition(C.count == Int(b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "inhib_sub_r3",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
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
    precondition(C.count == Int(b_), "C has wrong size")
    var n=n_
    var b=b_
    stream.dispatch(
        kernel: "inhib_div_r7",
        args: [
            .buffer(A.buffer),
            .buffer(W.buffer),
            .buffer(B.buffer),
            bytes(&n),
            bytes(&b)
        ],
        grid: MTLSize(
            width: Int(b)*((Int(n) + 127) / 128),
            height: 1,
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
            bytes(&n),
            bytes(&r),
            bytes(&lambda)
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
            bytes(&n),
            bytes(&w_NE),
            bytes(&w_ACh),
            bytes(&w_DA),
            bytes(&b),
            bytes(&eta_max)
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