
func runAllKernelSanityChecks() {
    print("Running full kernel sanity suite…")
    let tolerance: Float = 1e-3
    let relativeTolerance: Float = 1e-4

    func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Mismatched lengths")
        var diff: Float = 0
        for i in 0..<a.count {
            diff = max(diff, abs(a[i] - b[i]))
        }
        return diff
    }
    func reportArray(_ name: String, gpu: [Float], expected: [Float]) {
        guard gpu.count == expected.count else {
            print("❌ \(name) mismatched lengths gpu=\(gpu.count) expected=\(expected.count)")
            return
        }
        let diff = maxAbsDiff(gpu, expected)
        let status = diff <= tolerance ? "✅" : "❌"
        print("\(status) \(name) maxAbsDiff=\(diff)")
    }
    func reportScalar(_ name: String, gpu: Float, expected: Float) {
        let diff = abs(gpu - expected)
        let allowed = tolerance + relativeTolerance * max(abs(expected), 1)
        let status = diff <= allowed ? "✅" : "❌"
        print("\(status) \(name) absDiff=\(diff)")
    }
    func elementwise(_ A: [Float], _ B: [Float], op: (Float, Float) -> Float) -> [Float] {
        precondition(A.count == B.count)
        var out = [Float](repeating: 0, count: A.count)
        for i in 0..<A.count {
            out[i] = op(A[i], B[i])
        }
        return out
    }
    func cpuGemmRowMajor(A: [Float], B: [Float], m: Int, n: Int, p: Int, batches: Int) -> [Float] {
        var C = [Float](repeating: 0, count: m * p * batches)
        for batch in 0..<batches {
            let offsetA = batch * m * n
            let offsetB = batch * n * p
            let offsetC = batch * m * p
            for row in 0..<m {
                for col in 0..<p {
                    var acc: Float = 0
                    for k in 0..<n {
                        acc += A[offsetA + row * n + k] * B[offsetB + k * p + col]
                    }
                    C[offsetC + row * p + col] = acc
                }
            }
        }
        return C
    }
    func cpuGemmWithBTransposed(A: [Float], B: [Float], m: Int, n: Int, p: Int, batches: Int) -> [Float] {
        var C = [Float](repeating: 0, count: m * p * batches)
        for batch in 0..<batches {
            let offsetA = batch * m * n
            let offsetB = batch * p * n
            let offsetC = batch * m * p
            for row in 0..<m {
                for col in 0..<p {
                    var acc: Float = 0
                    for k in 0..<n {
                        acc += A[offsetA + row * n + k] * B[offsetB + col * n + k]
                    }
                    C[offsetC + row * p + col] = acc
                }
            }
        }
        return C
    }
    func cpuGemmWithATransposed(A: [Float], B: [Float], m: Int, n: Int, p: Int, batches: Int) -> [Float] {
        var C = [Float](repeating: 0, count: m * p * batches)
        for batch in 0..<batches {
            let offsetA = batch * m * n
            let offsetB = batch * n * p
            let offsetC = batch * m * p
            for row in 0..<m {
                for col in 0..<p {
                    var acc: Float = 0
                    for k in 0..<n {
                        acc += A[offsetA + k * m + row] * B[offsetB + k * p + col]
                    }
                    C[offsetC + row * p + col] = acc
                }
            }
        }
        return C
    }

    // Elementwise ops
    let ewN: UInt32 = 16
    let ewB: UInt32 = 2
    let totalEW = Int(ewN * ewB)
    let vecA = (0..<totalEW).map { Float($0) + 1 }
    let vecB = (0..<totalEW).map { Float($0) * 0.25 + 2 }

    var cAdd = [Float](repeating: 0, count: totalEW)
    add(A: vecA, B: vecB, C: &cAdd, n: ewN, b: ewB)
    reportArray("add", gpu: cAdd, expected: elementwise(vecA, vecB, op: +))

    var cSub = [Float](repeating: 0, count: totalEW)
    sub(A: vecA, B: vecB, C: &cSub, n: ewN, b: ewB)
    reportArray("sub", gpu: cSub, expected: elementwise(vecA, vecB, op: -))

    var cMul = [Float](repeating: 0, count: totalEW)
    mul(A: vecA, B: vecB, C: &cMul, n: ewN, b: ewB)
    reportArray("mul", gpu: cMul, expected: elementwise(vecA, vecB, op: *))

    var cDiv = [Float](repeating: 0, count: totalEW)
    div(A: vecA, B: vecB, C: &cDiv, n: ewN, b: ewB)
    reportArray("div", gpu: cDiv, expected: elementwise(vecA, vecB, op: /))

    // Embedding
    let vocab: UInt32 = 5
    let embedDim: UInt32 = 4
    let embedN: UInt32 = 3
    let embedTable = (0..<Int(vocab * embedDim)).map { Float($0) }
    let embedIdx: [UInt32] = [0, 2, 4]
    var embedOut = [Float](repeating: 0, count: Int(embedN * embedDim))
    embedding(A: embedTable, B: embedIdx, C: &embedOut, n: embedN, d: embedDim, vocab_size: vocab)
    var embedExpected = [Float](repeating: 0, count: embedOut.count)
    for i in 0..<Int(embedN) {
        let idx = Int(embedIdx[i])
        for j in 0..<Int(embedDim) {
            embedExpected[i * Int(embedDim) + j] = embedTable[idx * Int(embedDim) + j]
        }
    }
    reportArray("embedding", gpu: embedOut, expected: embedExpected)

    // GEMM family
    let gm: UInt32 = 32
    let gn: UInt32 = 32
    let gp: UInt32 = 32
    let gemm1A = (0..<Int(gm * gn)).map { Float(($0 % 17) - 8) }
    let gemm1B = (0..<Int(gn * gp)).map { Float(($0 % 13) - 6) }
    var gemm1C = [Float](repeating: 0, count: Int(gm * gp))
    gemm1(A: gemm1A, B: gemm1B, C: &gemm1C, m: gm, n: gn, p: gp, b: 1)
    let gemm1Expected = cpuGemmRowMajor(A: gemm1A, B: gemm1B, m: Int(gm), n: Int(gn), p: Int(gp), batches: 1)
    reportArray("gemm1", gpu: gemm1C, expected: gemm1Expected)

    let gemm2A = (0..<Int(gm * gn)).map { Float(($0 % 23) - 11) }
    let gemm2B = (0..<Int(gp * gn)).map { Float(($0 % 19) - 9) }
    var gemm2C = [Float](repeating: 0, count: Int(gm * gp))
    gemm2(A: gemm2A, B: gemm2B, C: &gemm2C, m: gm, n: gn, p: gp, b: 1)
    let gemm2Expected = cpuGemmWithBTransposed(A: gemm2A, B: gemm2B, m: Int(gm), n: Int(gn), p: Int(gp), batches: 1)
    reportArray("gemm2", gpu: gemm2C, expected: gemm2Expected)

    let gemm3A = (0..<Int(gn * gm)).map { Float(($0 % 29) - 14) }
    let gemm3B = (0..<Int(gn * gp)).map { Float(($0 % 31) - 7) }
    var gemm3C = [Float](repeating: 0, count: Int(gm * gp))
    gemm3(A: gemm3A, B: gemm3B, C: &gemm3C, m: gm, n: gn, p: gp, b: 1)
    let gemm3Expected = cpuGemmWithATransposed(A: gemm3A, B: gemm3B, m: Int(gm), n: Int(gn), p: Int(gp), batches: 1)
    reportArray("gemm3", gpu: gemm3C, expected: gemm3Expected)

    // Layer norm
    let lnN: UInt32 = 8
    let lnB: UInt32 = 2
    let lnInput = (0..<Int(lnN * lnB)).map { Float($0) - 4 }
    var mu = [Float](repeating: 0, count: Int(lnB))
    var sigma2 = [Float](repeating: 0, count: Int(lnB))
    for batch in 0..<Int(lnB) {
        var mean: Float = 0
        for i in 0..<Int(lnN) {
            mean += lnInput[batch * Int(lnN) + i]
        }
        mean /= Float(lnN)
        mu[batch] = mean
        var varVal: Float = 0
        for i in 0..<Int(lnN) {
            let diff = lnInput[batch * Int(lnN) + i] - mean
            varVal += diff * diff / Float(lnN)
        }
        sigma2[batch] = varVal
    }
    var lnOut = [Float](repeating: 0, count: lnInput.count)
    let eps: Float = 1e-5
    layernorm(A: lnInput, B: &lnOut, mu: mu, sigma2: sigma2, n: lnN, eps: eps, b: lnB)
    var lnExpected = [Float](repeating: 0, count: lnInput.count)
    for batch in 0..<Int(lnB) {
        let denom = sqrtf(sigma2[batch] + eps)
        for i in 0..<Int(lnN) {
            let idx = batch * Int(lnN) + i
            lnExpected[idx] = (lnInput[idx] - mu[batch]) / denom
        }
    }
    reportArray("layernorm", gpu: lnOut, expected: lnExpected)

    // Reductions (single batch)
    let redN: UInt32 = 256
#if true
    func reductionFinalValue(_ reduced: [Float]) -> Float {
        var cur = reduced
        while cur.count > 1 {
            let chunk = (cur.count + 127) / 128
            var next = [Float](repeating: 0, count: chunk)
            for i in 0..<chunk {
                var acc: Float = 0
                for j in 0..<128 {
                    let idx = i * 128 + j
                    if idx < cur.count {
                        acc += cur[idx]
                    }
                }
                next[i] = acc
            }
            cur = next
        }
        return cur.first ?? .nan
    }
#endif

    let redData = (0..<Int(redN)).map { Float(($0 % 37) - 18) }
    var maxOut = [Float](repeating: 0, count: (Int(redN)+127)/128)
    max_simd(A: redData, B: &maxOut, n: redN, b: 1)
    reportScalar("max_simd", gpu: maxOut.first ?? .nan, expected: redData.max() ?? .nan)

    var sumOut = [Float](repeating: 0, count: (Int(redN)+127)/128)
    sum_simd(A: redData, B: &sumOut, n: redN, b: 1)
    reportScalar("sum_simd", gpu: sumOut.first ?? .nan, expected: redData.reduce(0, +))

    var meanOut = [Float](repeating: 0, count: (Int(redN)+127)/128)
    mean_simd(A: redData, B: &meanOut, n: redN, b: 1)
    reportScalar("mean_simd", gpu: reductionFinalValue(meanOut), expected: redData.reduce(0, +) / Float(redData.count))

    let globalMax = redData.max() ?? 0
    var softmaxReduceOut = [Float](repeating: 0, count: (Int(redN)+127)/128)
    softmax_simd(A: redData, B: &softmaxReduceOut, global_max: [globalMax], n: redN, b: 1)
    let softmaxDenom = redData.map { expf($0 - globalMax) }.reduce(0, +)
    reportScalar("softmax_simd_reduce", gpu: reductionFinalValue(softmaxReduceOut), expected: softmaxDenom)

    let meanVal = redData.reduce(0, +) / Float(redData.count)
    var varianceOut = [Float](repeating: 0, count: (Int(redN)+127)/128)
    variance_simd(A: redData, B: &varianceOut, mu: [meanVal], n: redN, b: 1)
    let varianceExpected = redData.map { let diff = $0 - meanVal; return diff * diff }.reduce(0, +)
    reportScalar("variance_simd_reduce", gpu: reductionFinalValue(varianceOut), expected: varianceExpected)

    // Activations
    let actN: UInt32 = 16
    let actInput = (0..<Int(actN)).map { Float($0) - 8 }
    var tanhOut = actInput
    tanh(A: actInput, B: &tanhOut, n: actN, b: 1)
    let tanhExpected = actInput.map { tanhf($0) }
    reportArray("tanh", gpu: tanhOut, expected: tanhExpected)

    var reluOut = actInput
    relu(A: actInput, B: &reluOut, n: actN, b: 1)
    let reluExpected = actInput.map { max(0 as Float, $0) }
    reportArray("relu", gpu: reluOut, expected: reluExpected)

    // Softmax
    let smN: UInt32 = 4
    let smB: UInt32 = 2
    let softInput = (0..<Int(smN * smB)).map { Float(($0 % 5) - 2) }
    var softGlobalMax = [Float](repeating: 0, count: Int(smB))
    var softDenom = [Float](repeating: 0, count: Int(smB))
    for batch in 0..<Int(smB) {
        let start = batch * Int(smN)
        let slice = Array(softInput[start..<start + Int(smN)])
        let maxVal = slice.max() ?? 0
        softGlobalMax[batch] = maxVal
        softDenom[batch] = slice.map { expf($0 - maxVal) }.reduce(0, +)
    }
    var softOutput = [Float](repeating: 0, count: softInput.count)
    softmax(A: softInput, B: &softOutput, global_max: softGlobalMax, denom: softDenom, n: smN, b: smB)
    var softExpected = [Float](repeating: 0, count: softInput.count)
    for batch in 0..<Int(smB) {
        for i in 0..<Int(smN) {
            let idx = batch * Int(smN) + i
            softExpected[idx] = expf(softInput[idx] - softGlobalMax[batch]) / softDenom[batch]
        }
    }
    reportArray("softmax", gpu: softOutput, expected: softExpected)

    // Outer product
    let outerN: UInt32 = 32
    let outerM: UInt32 = 32
    let outerA = (0..<Int(outerN)).map { Float($0) + 1 }
    let outerBVec = (0..<Int(outerM)).map { Float($0) - 2 }
    var outerC = [Float](repeating: 0, count: Int(outerN * outerM))
    outer_prod(A: outerA, B: outerBVec, C: &outerC, n: outerN, m: outerM, b: 1)
    var outerExpected = [Float](repeating: 0, count: outerC.count)
    for i in 0..<Int(outerN) {
        for j in 0..<Int(outerM) {
            outerExpected[i * Int(outerM) + j] = outerA[i] * outerBVec[j]
        }
    }
    reportArray("outer_prod", gpu: outerC, expected: outerExpected)
}

runAllKernelSanityChecks()
