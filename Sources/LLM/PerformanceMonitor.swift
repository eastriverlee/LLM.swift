import Foundation

/// Performance monitoring session for tracking metrics over time.
public class PerformanceMonitor: ObservableObject {
    @Published public private(set) var currentMetrics: PerformanceMetrics?
    @Published public private(set) var isProfiling = false
    
    private var sessionStartTime: Date?
    private var sessionMetrics: [PerformanceMetrics] = []
    private var tokenCount: Int = 0
    private var startTime: Date?
    private var peakMemoryUsage: Int64 = 0
    
    // Operation timing properties moved from LLMCore
    private var operationStartTime: Date?
    private var tokensGeneratedInOperation: Int = 0
    private var modelLoadStartTime: Date?
    
    public init() {}
    
    /// Start a new profiling session
    public func startProfiling() {
        isProfiling = true
        sessionStartTime = Date()
        sessionMetrics.removeAll()
        tokenCount = 0
        peakMemoryUsage = 0
    }
    
    /// Stop profiling and return a performance report
    public func stopProfiling() -> PerformanceReport {
        isProfiling = false
        let report = PerformanceReport(
            sessionDuration: sessionStartTime.map { Date().timeIntervalSince($0) } ?? 0,
            metrics: sessionMetrics,
            averageTokensPerSecond: sessionMetrics.map { $0.tokensPerSecond }.reduce(0, +) / Double(max(sessionMetrics.count, 1)),
            totalTokensGenerated: sessionMetrics.map { $0.tokensGenerated }.reduce(0, +),
            peakMemoryUsage: peakMemoryUsage
        )
        sessionMetrics.removeAll()
        sessionStartTime = nil
        return report
    }
    
    /// Record metrics for a single operation
    func recordMetrics(_ metrics: PerformanceMetrics) {
        if isProfiling {
            sessionMetrics.append(metrics)
            peakMemoryUsage = max(peakMemoryUsage, metrics.peakMemoryUsage)
        }
        currentMetrics = metrics
    }
    
    /// Start timing an operation
    func startOperation() {
        startTime = Date()
        tokenCount = 0
        operationStartTime = Date()
        tokensGeneratedInOperation = 0
    }
    
    /// End timing an operation and record metrics
    func endOperation(tokensGenerated: Int, contextLength: Int) -> PerformanceMetrics {
        guard let startTime = startTime else {
            return PerformanceMetrics(
                tokensPerSecond: 0,
                memoryUsage: 0,
                inferenceTime: 0,
                contextLength: contextLength,
                tokensGenerated: tokensGenerated,
                averageTimePerToken: 0,
                peakMemoryUsage: 0
            )
        }
        
        let inferenceTime = Date().timeIntervalSince(startTime)
        let tokensPerSecond = inferenceTime > 0 ? Double(tokensGenerated) / inferenceTime : 0
        let averageTimePerToken = tokensGenerated > 0 ? inferenceTime / Double(tokensGenerated) : 0
        
        let metrics = PerformanceMetrics(
            tokensPerSecond: tokensPerSecond,
            memoryUsage: getCurrentMemoryUsage(),
            inferenceTime: inferenceTime,
            contextLength: contextLength,
            tokensGenerated: tokensGenerated,
            averageTimePerToken: averageTimePerToken,
            peakMemoryUsage: peakMemoryUsage
        )
        
        recordMetrics(metrics)
        return metrics
    }
    
    /// Start model loading timing
    func startModelLoad() {
        modelLoadStartTime = Date()
    }
    
    /// Record model load time and update metrics
    func recordModelLoadTime() {
        if let modelLoadStartTime = modelLoadStartTime {
            let loadTime = Date().timeIntervalSince(modelLoadStartTime)
            if let currentMetrics = currentMetrics {
                let updatedMetrics = PerformanceMetrics(
                    tokensPerSecond: currentMetrics.tokensPerSecond,
                    memoryUsage: currentMetrics.memoryUsage,
                    inferenceTime: currentMetrics.inferenceTime,
                    contextLength: currentMetrics.contextLength,
                    tokensGenerated: currentMetrics.tokensGenerated,
                    averageTimePerToken: currentMetrics.averageTimePerToken,
                    peakMemoryUsage: currentMetrics.peakMemoryUsage,
                    modelLoadTime: loadTime,
                    contextPrepTime: currentMetrics.contextPrepTime
                )
                recordMetrics(updatedMetrics)
            }
            self.modelLoadStartTime = nil
        }
    }
    
    /// Increment tokens generated in current operation
    func incrementTokensGenerated() {
        tokensGeneratedInOperation += 1
    }
    
    /// Get tokens generated in current operation
    func getTokensGeneratedInOperation() -> Int {
        return tokensGeneratedInOperation
    }
    
    /// Check if operation is active
    func isOperationActive() -> Bool {
        return operationStartTime != nil
    }
    
    /// Reset operation state
    func resetOperation() {
        operationStartTime = nil
        tokensGeneratedInOperation = 0
    }
    
    /// Get current memory usage
    private func getCurrentMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return kerr == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }
}
