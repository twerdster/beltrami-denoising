import Foundation
import Metal

public struct BeltramiMetalResult: Sendable {
    public let pixels: [Float]
    public let gpuMilliseconds: Double
    public let wallMilliseconds: Double
    public let temporaryBytes: Int
    public let tiled: Bool
    public let tileCount: Int
    public let commandBufferCount: Int
    public let temporalBatchIterations: Int
}

public enum BeltramiMetalError: LocalizedError {
    case unavailable(String)
    case invalidArgument(String)
    case gpuFailure(String)

    public var errorDescription: String? {
        switch self {
        case .unavailable(let message), .invalidArgument(let message), .gpuFailure(let message):
            return message
        }
    }
}

private struct BeltramiParameters {
    var width: UInt32
    var height: UInt32
    var radius: UInt32
    var reserved: UInt32 = 0
    var beta: Float
    var dt: Float
}

/// Float32 Patch-Beltrami flow, without resizing or clipping numerical values.
/// Calls are serialized; buffers are reused across iterations and invocations.
public final class BeltramiMetal: @unchecked Sendable {
    public let device: MTLDevice
    public let memoryBudgetBytes: Int
    private let queue: MTLCommandQueue
    private let metric: MTLComputePipelineState
    private let horizontal: MTLComputePipelineState
    private let vertical: MTLComputePipelineState
    private let local: MTLComputePipelineState
    private let update: MTLComputePipelineState
    private let lock = NSLock()
    private let group = MTLSize(width: 16, height: 8, depth: 1)
    private var workspace: Workspace?

    private final class Workspace {
        let capacity: Int
        let images: [MTLBuffer]
        let geometry: MTLBuffer
        let box: MTLBuffer?
        let fluxX: MTLBuffer
        let fluxY: MTLBuffer
        let bytes: Int

        init(device: MTLDevice, capacity: Int, needsBox: Bool) throws {
            self.capacity = capacity
            let length = capacity*MemoryLayout<SIMD4<Float>>.stride
            func buffer(_ name: String, shared: Bool = false) throws -> MTLBuffer {
                guard let result = device.makeBuffer(length: length,
                    options: shared ? .storageModeShared : .storageModePrivate) else {
                    throw BeltramiMetalError.unavailable("Could not allocate \(name) (\(length) bytes)")
                }
                result.label = name
                return result
            }
            images = try [buffer("Beltrami image A", shared: true), buffer("Beltrami image B", shared: true)]
            geometry = try buffer("Beltrami metric")
            box = needsBox ? try buffer("Beltrami horizontal sums") : nil
            fluxX = try buffer("Beltrami row flux")
            fluxY = try buffer("Beltrami column flux")
            bytes = length*(needsBox ? 6 : 5)
        }
    }

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice(), library: MTLLibrary? = nil,
                memoryBudgetBytes: Int = 192*1024*1024) throws {
        guard let device else { throw BeltramiMetalError.unavailable("Metal is unavailable") }
        guard memoryBudgetBytes > 0, let queue = device.makeCommandQueue() else {
            throw BeltramiMetalError.unavailable("A positive memory budget and a Metal command queue are required")
        }
        guard let library = library ?? device.makeDefaultLibrary() else {
            throw BeltramiMetalError.unavailable("Include BeltramiKernels.metal in the application target")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw BeltramiMetalError.unavailable("Missing Metal function \(name)")
            }
            return try device.makeComputePipelineState(function: function)
        }
        self.device = device
        self.queue = queue
        self.memoryBudgetBytes = memoryBudgetBytes
        metric = try pipeline("beltrami_metric")
        horizontal = try pipeline("beltrami_box_horizontal")
        vertical = try pipeline("beltrami_flux_vertical")
        local = try pipeline("beltrami_flux_local")
        update = try pipeline("beltrami_update")
    }

    private func allocate(_ count: Int, radius: Int) throws -> Workspace {
        if let workspace, workspace.capacity >= count, radius == 0 || workspace.box != nil {
            return workspace
        }
        workspace = nil
        let created = try Workspace(device: device, capacity: count, needsBox: radius > 0)
        workspace = created
        return created
    }

    /// Inputs and outputs are interleaved grayscale or RGB on the original 0–255 scale.
    /// Large images use exact dependency halos and temporal blocks, never resampling.
    public func denoise(_ pixels: [Float], width: Int, height: Int, channels: Int,
                        beta: Float, alpha: Float, iterations: Int, radius: Int) throws -> BeltramiMetalResult {
        lock.lock()
        defer { lock.unlock() }
        let started = ProcessInfo.processInfo.systemUptime
        guard width > 0, height > 0, width <= Int(Int32.max), height <= Int(Int32.max),
              channels == 1 || channels == 3,
              width <= Int.max/height, width*height <= Int.max/channels,
              pixels.count == width*height*channels else {
            throw BeltramiMetalError.invalidArgument("Expected width×height interleaved grayscale or RGB pixels")
        }
        guard beta.isFinite, beta > 0, alpha.isFinite, alpha >= 0,
              (beta*alpha).isFinite, iterations >= 0, radius >= 0, radius <= 128,
              pixels.allSatisfy({ $0.isFinite }) else {
            throw BeltramiMetalError.invalidArgument("Require finite pixels, beta>0, alpha>=0, iterations>=0 and radius0...128")
        }
        let horizontalBytes = (group.width+2*radius)*group.height*16
        let verticalBytes = group.width*(group.height+2*radius)*16
        guard max(horizontalBytes, verticalBytes) <= device.maxThreadgroupMemoryLength else {
            throw BeltramiMetalError.invalidArgument("Patch radius exceeds this device's threadgroup memory")
        }
        let bytesPerPixel = radius == 0 ? 80 : 96
        let maximumPixels = min(memoryBudgetBytes/bytesPerPixel, device.maxBufferLength/16)
        let count = width*height
        var gpuSeconds = 0.0
        var commands = 0
        var tiles = 0
        var temporalBatch = iterations
        var current = pixels
        let tiled = count > maximumPixels
        if iterations > 0 && alpha > 0 {
            if !tiled {
                let work = try allocate(count, radius: radius)
                gpuSeconds += try runTile(current, &current, imageWidth: width, channels: channels,
                    x0: 0, y0: 0, tileWidth: width, tileHeight: height,
                    coreX: 0, coreY: 0, coreWidth: width, coreHeight: height,
                    beta: beta, alpha: alpha, iterations: iterations, radius: radius, work: work)
                commands = 1
                tiles = 1
            } else {
                let side = Int(Double(maximumPixels).squareRoot())
                temporalBatch = min(iterations, min(16, max(1, side/(4*(radius+1)))))
                guard side > 2*(radius+1) else {
                    throw BeltramiMetalError.invalidArgument("Memory budget is too small for one dependency halo")
                }
                let work = try allocate(min(width, side)*min(height, side), radius: radius)
                var next = [Float](repeating: 0, count: pixels.count)
                var completed = 0
                while completed < iterations {
                    let batch = min(temporalBatch, iterations-completed)
                    let halo = batch*(radius+1)
                    let coreSide = side-2*halo
                    for coreY in stride(from: 0, to: height, by: coreSide) {
                        for coreX in stride(from: 0, to: width, by: coreSide) {
                            let coreWidth = min(coreSide, width-coreX)
                            let coreHeight = min(coreSide, height-coreY)
                            let x0 = max(0, coreX-halo), y0 = max(0, coreY-halo)
                            let x1 = min(width, coreX+coreWidth+halo)
                            let y1 = min(height, coreY+coreHeight+halo)
                            gpuSeconds += try runTile(current, &next, imageWidth: width, channels: channels,
                                x0: x0, y0: y0, tileWidth: x1-x0, tileHeight: y1-y0,
                                coreX: coreX, coreY: coreY, coreWidth: coreWidth, coreHeight: coreHeight,
                                beta: beta, alpha: alpha, iterations: batch, radius: radius, work: work)
                            commands += 1
                            tiles += 1
                        }
                    }
                    swap(&current, &next)
                    completed += batch
                }
            }
        }
        return BeltramiMetalResult(pixels: current, gpuMilliseconds: gpuSeconds*1000,
            wallMilliseconds: (ProcessInfo.processInfo.systemUptime-started)*1000,
            temporaryBytes: workspace?.bytes ?? 0, tiled: tiled, tileCount: tiles,
            commandBufferCount: commands, temporalBatchIterations: temporalBatch)
    }

    private func runTile(_ source: [Float], _ destination: inout [Float], imageWidth: Int, channels: Int,
                         x0: Int, y0: Int, tileWidth: Int, tileHeight: Int,
                         coreX: Int, coreY: Int, coreWidth: Int, coreHeight: Int,
                         beta: Float, alpha: Float, iterations: Int, radius: Int,
                         work: Workspace) throws -> Double {
        let packed = work.images[0].contents().bindMemory(to: SIMD4<Float>.self, capacity: work.capacity)
        source.withUnsafeBufferPointer { input in
            for y in 0..<tileHeight {
                for x in 0..<tileWidth {
                    let index = ((y+y0)*imageWidth+x+x0)*channels
                    packed[y*tileWidth+x] = channels == 1
                        ? SIMD4(input[index], 0, 0, 0)
                        : SIMD4(input[index], input[index+1], input[index+2], 0)
                }
            }
        }
        let gpuSeconds = try autoreleasepool { () throws -> Double in
            guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw BeltramiMetalError.unavailable("Could not encode the Beltrami flow")
            }
            command.label = "Patch Beltrami \(iterations) iterations"
            var parameters = BeltramiParameters(width: UInt32(tileWidth), height: UInt32(tileHeight),
                                               radius: UInt32(radius), beta: beta, dt: alpha*beta)
            let groups = MTLSize(width: (tileWidth+group.width-1)/group.width,
                                 height: (tileHeight+group.height-1)/group.height, depth: 1)
            func dispatch(_ pipeline: MTLComputePipelineState, buffers: [MTLBuffer], sharedBytes: Int = 0) {
                encoder.setComputePipelineState(pipeline)
                for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
                encoder.setBytes(&parameters, length: MemoryLayout<BeltramiParameters>.stride, index: buffers.count)
                if sharedBytes > 0 { encoder.setThreadgroupMemoryLength(sharedBytes, index: 0) }
                encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: group)
                encoder.memoryBarrier(scope: .buffers)
            }
            for iteration in 0..<iterations {
                let input = work.images[iteration%2], output = work.images[(iteration+1)%2]
                if radius == 0 {
                    dispatch(local, buffers: [input, work.geometry, work.fluxX, work.fluxY])
                } else {
                    dispatch(metric, buffers: [input, work.geometry])
                    dispatch(horizontal, buffers: [work.geometry, work.box!],
                             sharedBytes: (group.width+2*radius)*group.height*16)
                    dispatch(vertical, buffers: [input, work.box!, work.geometry, work.fluxX, work.fluxY],
                             sharedBytes: group.width*(group.height+2*radius)*16)
                }
                dispatch(update, buffers: [input, work.geometry, work.fluxX, work.fluxY, output])
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                throw BeltramiMetalError.gpuFailure(command.error?.localizedDescription ?? "Metal execution failed")
            }
            return max(0, command.gpuEndTime-command.gpuStartTime)
        }
        let output = work.images[iterations%2].contents().bindMemory(to: SIMD4<Float>.self, capacity: work.capacity)
        try destination.withUnsafeMutableBufferPointer { target in
            for y in 0..<coreHeight {
                for x in 0..<coreWidth {
                    let value = output[(coreY-y0+y)*tileWidth+coreX-x0+x]
                    let index = ((coreY+y)*imageWidth+coreX+x)*channels
                    guard value.x.isFinite, channels == 1 || value.y.isFinite && value.z.isFinite else {
                        throw BeltramiMetalError.gpuFailure("Non-finite flow value; check beta, alpha and image intensity scale")
                    }
                    target[index] = value.x
                    if channels == 3 { target[index+1] = value.y; target[index+2] = value.z }
                }
            }
        }
        return gpuSeconds
    }
}
