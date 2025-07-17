import Testing
import Foundation
@testable import LLM

@Suite("PerformanceMonitorTests")
struct PerformanceMonitorTests {
    // MARK: Performance Monitoring Tests
    let systemPrompt = "You are a helpful AI assistant."
    lazy var model = HuggingFaceModel("unsloth/Qwen3-0.6B-GGUF", .Q4_K_M, template: .chatML(systemPrompt))
    //let urlString = "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf?download=true"
  
    @Test
    func testPerformanceMonitorInitialization() throws {
        let monitor = PerformanceMonitor()
        #expect(monitor.isProfiling == false)
        #expect(monitor.currentMetrics == nil)
    }
    
    @Test
    func testPerformanceMonitorStartStop() throws {
        let monitor = PerformanceMonitor()
        
        monitor.startProfiling()
        #expect(monitor.isProfiling == true)
        
        let report = monitor.stopProfiling()
        #expect(monitor.isProfiling == false)
        #expect(report.sessionDuration >= 0)
        #expect(report.metrics.isEmpty)
        #expect(report.averageTokensPerSecond == 0)
        #expect(report.totalTokensGenerated == 0)
        #expect(report.peakMemoryUsage == 0)
    }
    
    @Test
    func testPerformanceMetricsCreation() throws {
        let metrics = PerformanceMetrics(
            tokensPerSecond: 10.5,
            memoryUsage: 1024 * 1024, // 1MB
            inferenceTime: 2.5,
            contextLength: 100,
            tokensGenerated: 25,
            averageTimePerToken: 0.1,
            peakMemoryUsage: 2 * 1024 * 1024, // 2MB
            modelLoadTime: 1.0,
            contextPrepTime: 0.1
        )
        
        #expect(metrics.tokensPerSecond == 10.5)
        #expect(metrics.memoryUsage == 1024 * 1024)
        #expect(metrics.inferenceTime == 2.5)
        #expect(metrics.contextLength == 100)
        #expect(metrics.tokensGenerated == 25)
        #expect(metrics.averageTimePerToken == 0.1)
        #expect(metrics.peakMemoryUsage == 2 * 1024 * 1024)
        #expect(metrics.modelLoadTime == 1.0)
        #expect(metrics.contextPrepTime == 0.1)
    }
    
    @Test
    func testPerformanceReportCalculations() throws {
        let metrics1 = PerformanceMetrics(
            tokensPerSecond: 10.0,
            memoryUsage: 1024 * 1024,
            inferenceTime: 1.0,
            contextLength: 50,
            tokensGenerated: 10,
            averageTimePerToken: 0.1,
            peakMemoryUsage: 1024 * 1024
        )
        
        let metrics2 = PerformanceMetrics(
            tokensPerSecond: 20.0,
            memoryUsage: 2 * 1024 * 1024,
            inferenceTime: 2.0,
            contextLength: 100,
            tokensGenerated: 40,
            averageTimePerToken: 0.05,
            peakMemoryUsage: 2 * 1024 * 1024
        )
        
        let report = PerformanceReport(
            sessionDuration: 5.0,
            metrics: [metrics1, metrics2],
            averageTokensPerSecond: 15.0,
            totalTokensGenerated: 50,
            peakMemoryUsage: 2 * 1024 * 1024
        )
        
        #expect(report.sessionDuration == 5.0)
        #expect(report.metrics.count == 2)
        #expect(report.averageTokensPerSecond == 15.0)
        #expect(report.totalTokensGenerated == 50)
        #expect(report.peakMemoryUsage == 2 * 1024 * 1024)
        #expect(abs(report.averageInferenceTime - 1.5) < 0.001)
        let expectedMemory: Int64 = Int64(1.5 * 1024 * 1024)
        let diff = abs(report.averageMemoryUsage - expectedMemory)
        #expect(diff < 1)
        #expect(report.bestMetrics?.tokensPerSecond == 20.0)
        #expect(report.worstMetrics?.tokensPerSecond == 10.0)
    }
    
    @Test
    mutating func testLLMWithPerformanceMonitoring() async throws {
        let bot = try await LLM(from: model)!
        
        // Start profiling
        await bot.startProfiling()
        #expect(bot.performanceMonitor.isProfiling == true)
        
        // Perform some operations
        let input = "Generate a short response about AI."
        await bot.respond(to: input)
        
        // Check that we have metrics
        let metrics = await bot.getPerformanceMetrics()
        #expect(metrics != nil)
        if let metrics = metrics {
            #expect(metrics.tokensGenerated > 0)
            #expect(metrics.inferenceTime > 0)
            #expect(metrics.contextLength > 0)
            #expect(metrics.memoryUsage > 0)
        }
        
        // Stop profiling and get report
        let report = await bot.stopProfiling()
        #expect(report != nil)
        #expect(bot.performanceMonitor.isProfiling == false)
        
        if let report = report {
            #expect(report.metrics.count > 0)
            #expect(report.totalTokensGenerated > 0)
            #expect(report.sessionDuration > 0)
        }
    }
    
    @Test
    mutating func testPerformanceMonitoringWithMultipleOperations() async throws {
        let bot = try await LLM(from: model)!
        
        await bot.startProfiling()
        
        // Perform multiple operations
        let inputs = [
            "What is AI?",
            "Explain machine learning.",
            "Tell me about neural networks."
        ]
        
        for input in inputs {
            await bot.respond(to: input)
        }
        
        let report = await bot.stopProfiling()
        #expect(report != nil)
        
        if let report = report {
            #expect(report.metrics.count >= 3)
            #expect(report.totalTokensGenerated > 0)
            #expect(report.averageTokensPerSecond > 0)
            
            // Check that we have both best and worst metrics
            #expect(report.bestMetrics != nil)
            #expect(report.worstMetrics != nil)
            
            if let best = report.bestMetrics, let worst = report.worstMetrics {
                #expect(best.tokensPerSecond >= worst.tokensPerSecond)
            }
        }
    }
    
    @Test
    mutating func testPerformanceMetricsPersistence() async throws {
        let bot = try await LLM(from: model)!
        
        // Perform operation without profiling
        await bot.respond(to: "Hello")
        
        // Check that we don't get metrics when not profiling
        let metrics = await bot.getPerformanceMetrics()
        #expect(metrics == nil)
        
        // Start profiling and perform another operation
        await bot.startProfiling()
        await bot.respond(to: "World")
        
        // Check that we get metrics during profiling
        let profilingMetrics = await bot.getPerformanceMetrics()
        #expect(profilingMetrics != nil)
        
        let report = await bot.stopProfiling()
        #expect(report != nil)
        #expect(report?.metrics.count == 1)
    }
}
