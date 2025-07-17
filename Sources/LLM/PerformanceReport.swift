import Foundation

/// Performance report containing aggregated metrics from a profiling session.
public struct PerformanceReport: Codable {
    /// Total duration of the profiling session
    public let sessionDuration: TimeInterval
    /// All recorded metrics during the session
    public let metrics: [PerformanceMetrics]
    /// Average tokens per second across all operations
    public let averageTokensPerSecond: Double
    /// Total tokens generated during the session
    public let totalTokensGenerated: Int
    /// Peak memory usage during the session
    public let peakMemoryUsage: Int64
    
    /// Get the best performance metrics from the session
    public var bestMetrics: PerformanceMetrics? {
        metrics.max { $0.tokensPerSecond < $1.tokensPerSecond }
    }
    
    /// Get the worst performance metrics from the session
    public var worstMetrics: PerformanceMetrics? {
        metrics.min { $0.tokensPerSecond < $1.tokensPerSecond }
    }
    
    /// Average inference time across all operations
    public var averageInferenceTime: TimeInterval {
        guard !metrics.isEmpty else { return 0 }
        return metrics.map { $0.inferenceTime }.reduce(0, +) / Double(metrics.count)
    }
    
    /// Average memory usage across all operations
    public var averageMemoryUsage: Int64 {
        guard !metrics.isEmpty else { return 0 }
        return metrics.map { $0.memoryUsage }.reduce(0, +) / Int64(metrics.count)
    }
}
