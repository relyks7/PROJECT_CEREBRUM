
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
    func deviceBuffer(_ values: [Float]) -> DeviceFloatBuffer {
        DeviceFloatBuffer(values)
    }
    func zerosBuffer(_ count: Int) -> DeviceFloatBuffer {
        let buf = DeviceFloatBuffer(count: count)
        buf.fill(0)
        return buf
    }
    func readBuffer(_ buffer: DeviceFloatBuffer) -> [Float] {
        buffer.toArray()
    }

    // Elementwise ops
    let ewN: UInt32 = 16
    let ewB: UInt32 = 2
    let totalEW = Int(ewN * ewB)
    let vecA = (0..<totalEW).map { Float($0) + 1 }
    let vecB = (0..<totalEW).map { Float($0) * 0.25 + 2 }

    let ewBufA = deviceBuffer(vecA)
    let ewBufB = deviceBuffer(vecB)

    let addOut = zerosBuffer(totalEW)
    add(ewBufA, ewBufB, addOut, ewN, ewB)
    reportArray("add", gpu: readBuffer(addOut), expected: elementwise(vecA, vecB, op: +))

    let subOut = zerosBuffer(totalEW)
    sub(ewBufA, ewBufB, subOut, ewN, ewB)
    reportArray("sub", gpu: readBuffer(subOut), expected: elementwise(vecA, vecB, op: -))

    let mulOut = zerosBuffer(totalEW)
    mul(ewBufA, ewBufB, mulOut, ewN, ewB)
    reportArray("mul", gpu: readBuffer(mulOut), expected: elementwise(vecA, vecB, op: *))

    let divOut = zerosBuffer(totalEW)
    div(ewBufA, ewBufB, divOut, ewN, ewB)
    reportArray("div", gpu: readBuffer(divOut), expected: elementwise(vecA, vecB, op: /))

    // Embedding
    let vocab: UInt32 = 5
    let embedDim: UInt32 = 4
    let embedN: UInt32 = 3
    let embedTable = (0..<Int(vocab * embedDim)).map { Float($0) }
    let embedIdx: [UInt32] = [0, 2, 4]
    let embedTableBuf = deviceBuffer(embedTable)
    let embedOutBuf = zerosBuffer(Int(embedN * embedDim))
    embedding(embedTableBuf, embedIdx, embedOutBuf, embedN, embedDim, vocab)
    let embedOut = readBuffer(embedOutBuf)
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
    let gemm1ABuf = deviceBuffer(gemm1A)
    let gemm1BBuf = deviceBuffer(gemm1B)
    let gemm1Cbuf = zerosBuffer(Int(gm * gp))
    gemm1(gemm1ABuf, gemm1BBuf, gemm1Cbuf, gm, gn, gp, 1)
    let gemm1Expected = cpuGemmRowMajor(A: gemm1A, B: gemm1B, m: Int(gm), n: Int(gn), p: Int(gp), batches: 1)
    reportArray("gemm1", gpu: readBuffer(gemm1Cbuf), expected: gemm1Expected)

    let gemm2A = (0..<Int(gm * gn)).map { Float(($0 % 23) - 11) }
    let gemm2B = (0..<Int(gp * gn)).map { Float(($0 % 19) - 9) }
    let gemm2ABuf = deviceBuffer(gemm2A)
    let gemm2BBuf = deviceBuffer(gemm2B)
    let gemm2Cbuf = zerosBuffer(Int(gm * gp))
    gemm2(gemm2ABuf, gemm2BBuf, gemm2Cbuf, gm, gn, gp, 1)
    let gemm2Expected = cpuGemmWithBTransposed(A: gemm2A, B: gemm2B, m: Int(gm), n: Int(gn), p: Int(gp), batches: 1)
    reportArray("gemm2", gpu: readBuffer(gemm2Cbuf), expected: gemm2Expected)

    let gemm3A = (0..<Int(gn * gm)).map { Float(($0 % 29) - 14) }
    let gemm3B = (0..<Int(gn * gp)).map { Float(($0 % 31) - 7) }
    let gemm3ABuf = deviceBuffer(gemm3A)
    let gemm3BBuf = deviceBuffer(gemm3B)
    let gemm3Cbuf = zerosBuffer(Int(gm * gp))
    gemm3(gemm3ABuf, gemm3BBuf, gemm3Cbuf, gm, gn, gp, 1)
    let gemm3Expected = cpuGemmWithATransposed(A: gemm3A, B: gemm3B, m: Int(gm), n: Int(gn), p: Int(gp), batches: 1)
    reportArray("gemm3", gpu: readBuffer(gemm3Cbuf), expected: gemm3Expected)

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
    let eps: Float = 1e-5
    let lnInBuf = deviceBuffer(lnInput)
    let lnOutBuf = zerosBuffer(lnInput.count)
    let muBuf = deviceBuffer(mu)
    let sigmaBuf = deviceBuffer(sigma2)
    layernorm(lnInBuf, lnOutBuf, muBuf, sigmaBuf, lnN, eps, lnB)
    let lnOut = readBuffer(lnOutBuf)
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
    let redData = (0..<Int(redN)).map { Float(($0 % 37) - 18) }
    let redBuf = deviceBuffer(redData)

    let maxOutBuf = zerosBuffer(1)
    max_simd(redBuf, maxOutBuf, redN, 1)
    reportScalar("max_simd", gpu: readBuffer(maxOutBuf).first ?? .nan, expected: redData.max() ?? .nan)

    let sumOutBuf = zerosBuffer(1)
    sum_simd(redBuf, sumOutBuf, redN, 1)
    reportScalar("sum_simd", gpu: readBuffer(sumOutBuf).first ?? .nan, expected: redData.reduce(0, +))

    let meanOutBuf = zerosBuffer(1)
    mean_simd(redBuf, meanOutBuf, redN, 1)
    reportScalar("mean_simd", gpu: readBuffer(meanOutBuf).first ?? .nan, expected: redData.reduce(0, +) / Float(redData.count))

    let globalMax = redData.max() ?? 0
    let softmaxReduceBuf = zerosBuffer(1)
    let globalMaxBuf = deviceBuffer([globalMax])
    softmax_simd(redBuf, softmaxReduceBuf, globalMaxBuf, redN, 1)
    let softmaxDenom = redData.map { expf($0 - globalMax) }.reduce(0, +)
    reportScalar("softmax_simd_reduce", gpu: readBuffer(softmaxReduceBuf).first ?? .nan, expected: softmaxDenom)

    let meanVal = redData.reduce(0, +) / Float(redData.count)
    let varianceOutBuf = zerosBuffer(1)
    let muBuf = deviceBuffer([meanVal])
    variance_simd(redBuf, varianceOutBuf, muBuf, redN, 1)
    let varianceExpected = redData.map { let diff = $0 - meanVal; return diff * diff }.reduce(0, +)
    reportScalar("variance_simd_reduce", gpu: readBuffer(varianceOutBuf).first ?? .nan, expected: varianceExpected)

    // Activations
    let actN: UInt32 = 16
    let actInput = (0..<Int(actN)).map { Float($0) - 8 }
    let actBuf = deviceBuffer(actInput)
    let tanhBuf = zerosBuffer(Int(actN))
    tanh(actBuf, tanhBuf, actN, 1)
    let tanhExpected = actInput.map { tanhf($0) }
    reportArray("tanh", gpu: readBuffer(tanhBuf), expected: tanhExpected)

    let reluBuf = zerosBuffer(Int(actN))
    relu(actBuf, reluBuf, actN, 1)
    let reluExpected = actInput.map { max(0 as Float, $0) }
    reportArray("relu", gpu: readBuffer(reluBuf), expected: reluExpected)

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
    let softInputBuf = deviceBuffer(softInput)
    let softOutputBuf = zerosBuffer(softInput.count)
    let softGlobalBuf = deviceBuffer(softGlobalMax)
    let softDenomBuf = deviceBuffer(softDenom)
    softmax(softInputBuf, softOutputBuf, softGlobalBuf, softDenomBuf, smN, smB)
    var softExpected = [Float](repeating: 0, count: softInput.count)
    for batch in 0..<Int(smB) {
        for i in 0..<Int(smN) {
            let idx = batch * Int(smN) + i
            softExpected[idx] = expf(softInput[idx] - softGlobalMax[batch]) / softDenom[batch]
        }
    }
    reportArray("softmax", gpu: readBuffer(softOutputBuf), expected: softExpected)

    // Outer product
    let outerN: UInt32 = 32
    let outerM: UInt32 = 32
    let outerA = (0..<Int(outerN)).map { Float($0) + 1 }
    let outerBVec = (0..<Int(outerM)).map { Float($0) - 2 }
    let outerABuf = deviceBuffer(outerA)
    let outerBBuf = deviceBuffer(outerBVec)
    let outerCBuf = zerosBuffer(Int(outerN * outerM))
    outer_prod(outerABuf, outerBBuf, outerCBuf, outerN, outerM, 1)
    var outerExpected = [Float](repeating: 0, count: Int(outerN * outerM))
    for i in 0..<Int(outerN) {
        for j in 0..<Int(outerM) {
            outerExpected[i * Int(outerM) + j] = outerA[i] * outerBVec[j]
        }
    }
    reportArray("outer_prod", gpu: readBuffer(outerCBuf), expected: outerExpected)
}

runAllKernelSanityChecks()
