import Foundation
import Metal

struct Fixture: Decodable {
    let name: String
    let width: Int
    let height: Int
    let channels: Int
    let radius: Int
    let iterations: Int
    let beta: Float
    let alpha: Float
    let input: String
    let expected: String
    let forceTiling: Bool
}

func option(_ name: String) -> String? {
    guard let index = CommandLine.arguments.firstIndex(of: name), index+1 < CommandLine.arguments.count else { return nil }
    return CommandLine.arguments[index+1]
}

func floats(_ path: URL) throws -> [Float] {
    let data = try Data(contentsOf: path)
    precondition(data.count%4 == 0)
    var values = [Float](repeating: 0, count: data.count/4)
    _ = values.withUnsafeMutableBytes { data.copyBytes(to: $0) }
    return values
}

func differences(_ actual: [Float], _ expected: [Float]) -> (Double, Double) {
    precondition(actual.count == expected.count)
    var maximum = 0.0, sum = 0.0
    for i in actual.indices {
        let error = Double(actual[i])-Double(expected[i])
        maximum = max(maximum, abs(error))
        sum += error*error
    }
    return (maximum, sqrt(sum/Double(actual.count)))
}

do {
    guard let path = option("--library"), let device = MTLCreateSystemDefaultDevice() else {
        throw BeltramiMetalError.invalidArgument("Use --library compiled.metallib [--fixtures directory] [--benchmarks] [--output file]")
    }
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: path))
    let engine = try BeltramiMetal(device: device, library: library)
    var verification: [[String: Any]] = []
    var benchmarks: [[String: Any]] = []
    var passed = true
    if let fixtures = option("--fixtures") {
        let directory = URL(fileURLWithPath: fixtures)
        let cases = try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        for fixture in cases {
            let source = try floats(directory.appendingPathComponent(fixture.input))
            let expected = try floats(directory.appendingPathComponent(fixture.expected))
            let result = try engine.denoise(source, width: fixture.width, height: fixture.height,
                channels: fixture.channels, beta: fixture.beta, alpha: fixture.alpha,
                iterations: fixture.iterations, radius: fixture.radius)
            let (maximum, rms) = differences(result.pixels, expected)
            let finite = result.pixels.allSatisfy { $0.isFinite }
            let success = maximum < 0.01 && rms < 0.001 && finite
            var row: [String: Any] = ["name": fixture.name, "width": fixture.width, "height": fixture.height,
                "channels": fixture.channels, "radius": fixture.radius, "iterations": fixture.iterations,
                "maximumAbsoluteError": maximum, "rmsError": rms, "finite": finite,
                "gpuMilliseconds": result.gpuMilliseconds, "passed": success]
            if fixture.forceTiling {
                let tiled = try BeltramiMetal(device: device, library: library, memoryBudgetBytes: 64*1024)
                let output = try tiled.denoise(source, width: fixture.width, height: fixture.height,
                    channels: fixture.channels, beta: fixture.beta, alpha: fixture.alpha,
                    iterations: fixture.iterations, radius: fixture.radius)
                let (error, tileRMS) = differences(output.pixels, result.pixels)
                let same = output.pixels == result.pixels
                row["tiledMaximumAbsoluteError"] = error
                row["tiledRmsError"] = tileRMS
                row["tiledBitIdentical"] = same
                row["tileCount"] = output.tileCount
                row["temporaryBytes"] = output.temporaryBytes
                row["temporalBatchIterations"] = output.temporalBatchIterations
                if !same || !output.tiled || output.temporaryBytes > 64*1024 { passed = false }
            }
            verification.append(row)
            if !success { passed = false }
            print(String(data: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), encoding: .utf8)!)
        }
    }
    if CommandLine.arguments.contains("--benchmarks") {
        let settings = [(256, 256, 1, 2, 150), (512, 512, 1, 2, 150),
                        (512, 512, 3, 3, 10), (641, 479, 3, 2, 10),
                        (1920, 1080, 3, 3, 10), (512, 512, 3, 0, 10),
                        (4032, 3024, 1, 2, 5)]
        for (width, height, channels, radius, iterations) in settings {
            let source = (0..<width*height*channels).map { Float(($0*37)%251) }
            _ = try engine.denoise(source, width: width, height: height, channels: channels,
                beta: 800, alpha: 0.21, iterations: 2, radius: radius)
            var times = [Double](), walls = [Double]()
            var last: BeltramiMetalResult?
            for _ in 0..<3 {
                let output = try engine.denoise(source, width: width, height: height,
                    channels: channels, beta: 800, alpha: 0.21, iterations: iterations, radius: radius)
                times.append(output.gpuMilliseconds)
                walls.append(output.wallMilliseconds)
                last = output
            }
            let result = last!
            let row: [String: Any] = ["width": width, "height": height, "channels": channels,
                "iterations": iterations, "radius": radius, "gpuMilliseconds": times,
                "gpuMedianMilliseconds": times.sorted()[1], "wallMilliseconds": walls,
                "wallMedianMilliseconds": walls.sorted()[1], "temporaryBytes": result.temporaryBytes,
                "tiled": result.tiled, "tileCount": result.tileCount,
                "commandBufferCount": result.commandBufferCount,
                "temporalBatchIterations": result.temporalBatchIterations]
            benchmarks.append(row)
            print(String(data: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), encoding: .utf8)!)
        }
    }
    let report: [String: Any] = ["device": device.name, "backend": "Metal float32",
        "reference": "Python efficient variant, single precision, direct conv2",
        "maximumAbsoluteTolerance": 0.01, "rmsTolerance": 0.001,
        "verification": verification, "benchmarks": benchmarks, "passed": passed]
    if let output = option("--output") {
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: output))
    }
    if !passed { exit(1) }
} catch {
    fputs("\(error.localizedDescription)\n", stderr)
    exit(1)
}
