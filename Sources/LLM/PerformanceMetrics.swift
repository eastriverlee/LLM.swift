import Foundation

/// Performance metrics for LLM operations.
public struct PerformanceMetrics: Codable, Equatable {
    /// Tokens generated per second
    public let tokensPerSecond: Double
    /// Memory usage in bytes
    public let memoryUsage: Int64
    /// Total inference time in seconds
    public let inferenceTime: TimeInterval
    /// Current context length (number of tokens)
    public let contextLength: Int
    /// Number of tokens generated in the last operation
    public let tokensGenerated: Int
    /// Average time per token in milliseconds
    public let averageTimePerToken: TimeInterval
    /// Peak memory usage during operation
    public let peakMemoryUsage: Int64
    /// Model loading time in seconds
    public let modelLoadTime: TimeInterval?
    /// Context preparation time in seconds
    public let contextPrepTime: TimeInterval?
    
    public init(
        tokensPerSecond: Double,
        memoryUsage: Int64,
        inferenceTime: TimeInterval,
        contextLength: Int,
        tokensGenerated: Int,
        averageTimePerToken: TimeInterval,
        peakMemoryUsage: Int64,
        modelLoadTime: TimeInterval? = nil,
        contextPrepTime: TimeInterval? = nil
    ) {
        self.tokensPerSecond = tokensPerSecond
        self.memoryUsage = memoryUsage
        self.inferenceTime = inferenceTime
        self.contextLength = contextLength
        self.tokensGenerated = tokensGenerated
        self.averageTimePerToken = averageTimePerToken
        self.peakMemoryUsage = peakMemoryUsage
        self.modelLoadTime = modelLoadTime
        self.contextPrepTime = contextPrepTime
    }
}
