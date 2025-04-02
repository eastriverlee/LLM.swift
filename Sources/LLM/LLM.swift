import Foundation
import llama

public typealias Token = llama_token
public typealias Batch = llama_batch
public typealias Model = OpaquePointer
public typealias Vocab = OpaquePointer
public typealias Chat = (role: Role, content: String)

public actor LLMCore {
    private let model: Model
    private let vocab: Vocab
    private var context: OpaquePointer
    private var batch: llama_batch
    private let params: llama_context_params
    
    private(set) var seed: UInt32
    private(set) var topK: Int32
    private(set) var topP: Float
    private(set) var temp: Float
    
    private let maxTokenCount: Int
    private let totalTokenCount: Int
    private lazy var newlineToken: Token = llama_vocab_nl(vocab)
    private lazy var endToken: Token = llama_vocab_eos(vocab)
    private lazy var nullToken: Token = encode("\0", shouldAddBOS: false).first!
    
    private var stopSequenceTokens: [Token]?
    private var tokenBuffer: [Token] = []
    private let tokenDecodeCache = NSCache<NSNumber, NSString>()
    
    private var shouldContinuePredicting = false
    private var currentTokenCount: Int32 = 0
    
    func setParameters(seed: UInt32? = nil, topK: Int32? = nil, topP: Float? = nil, temp: Float? = nil) {
        if let seed { self.seed = seed }
        if let topK { self.topK = topK }
        if let topP { self.topP = topP }
        if let temp { self.temp = temp }
    }
    
    func setStopSequence(_ sequence: String?) {
        if let sequence {
            stopSequenceTokens = encode(sequence, shouldAddBOS: false)
        } else {
            stopSequenceTokens = nil
        }
    }
    
    public init(model: Model, path: [CChar], seed: UInt32, topK: Int32, topP: Float, temp: Float, maxTokenCount: Int) throws {
        self.model = model
        self.vocab = llama_model_get_vocab(model)
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temp = temp
        self.maxTokenCount = maxTokenCount
        self.totalTokenCount = Int(llama_vocab_n_tokens(vocab))
        
        var contextParams = llama_context_default_params()
        let processorCount = Int32(ProcessInfo().processorCount)
        contextParams.n_ctx = UInt32(maxTokenCount)
        contextParams.n_batch = contextParams.n_ctx
        contextParams.n_threads = processorCount
        contextParams.n_threads_batch = processorCount
        self.params = contextParams
        
        guard let context = llama_init_from_model(model, params) else {
            throw LLMError.contextCreationFailed
        }
        self.context = context
        
        self.batch = llama_batch_init(Int32(maxTokenCount), 0, 1)
    }
    
    deinit {
        llama_batch_free(batch)
        llama_free(context)
    }
    
    
    public func encode(_ text: String, shouldAddBOS: Bool = true) -> [Token] {
        let count = Int32(text.cString(using: .utf8)!.count)
        var tokenCount = count + 1
        let cTokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(tokenCount))
        defer { cTokens.deallocate() }
        
        tokenCount = llama_tokenize(vocab, text, count, cTokens, tokenCount, shouldAddBOS, true)
        let tokens = (0..<Int(tokenCount)).map { cTokens[$0] }
        return tokens
    }
    
    public func decode(_ token: Token) -> String {
        if let cached = tokenDecodeCache.object(forKey: NSNumber(value: token)) {
            return cached as String
        }
        
        var bufferLength = 16
        var buffer: [CChar] = .init(repeating: 0, count: bufferLength)
        var actualLength = Int(llama_token_to_piece(vocab, token, &buffer, Int32(bufferLength), 0, false))
        
        guard actualLength != 0 else { return "" }
        
        if actualLength < 0 {
            bufferLength = -actualLength
            buffer = .init(repeating: 0, count: bufferLength)
            actualLength = Int(llama_token_to_piece(vocab, token, &buffer, Int32(bufferLength), 0, false))
        } else {
            buffer = Array(buffer.prefix(actualLength))
        }
        
        let bytes = buffer.map { UInt8(bitPattern: $0) }
        guard let decoded = String(bytes: bytes, encoding: .utf8) else { return "" }
        
        tokenDecodeCache.setObject(decoded as NSString, forKey: NSNumber(value: token))
        
        return decoded
    }
    
    
    func prepareContext(for input: String) -> Bool {
        guard !input.isEmpty else { return false }
        
        tokenBuffer.removeAll()
        
        var tokens = encode(input)
        if tokens.last == nullToken { tokens.removeLast() }
        
        let initialCount = tokens.count
        guard maxTokenCount > initialCount else { return false }
        
        clearBatch()
        for (i, token) in tokens.enumerated() {
            addToBatch(token: token, pos: Int32(i), isLogit: i == initialCount - 1)
        }
        guard llama_decode(context, batch) == 0 else { return false }
        
        currentTokenCount = Int32(initialCount)
        shouldContinuePredicting = true
        return true
    }
    
    private func clearBatch() {
        batch.n_tokens = 0
    }
    
    private func addToBatch(token: Token, pos: Int32, isLogit: Bool = true) {
        let i = Int(batch.n_tokens)
        batch.token[i] = token
        batch.pos[i] = pos
        batch.n_seq_id[i] = 1
        if let seq_id = batch.seq_id[i] {
            seq_id[0] = 0
        }
        batch.logits[i] = isLogit ? 1 : 0
        batch.n_tokens += 1
    }
    
    func predictNextToken() -> Token {
        guard shouldContinuePredicting, currentTokenCount < Int32(maxTokenCount) else { 
            return endToken 
        }
        
        let samplerParams = llama_sampler_chain_default_params()
        let sampler = llama_sampler_chain_init(samplerParams)
        defer { llama_sampler_free(sampler) }
        
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK))
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1))
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed))
        
        let i = batch.n_tokens - 1
        let token = llama_sampler_sample(sampler, context, i)
        
        tokenBuffer.append(token)
        
        if let stopTokens = stopSequenceTokens, stopTokens.count <= tokenBuffer.count {
            let startIdx = tokenBuffer.count - stopTokens.count
            let bufferSlice = tokenBuffer[startIdx..<tokenBuffer.count]
            if Array(bufferSlice) == stopTokens {
                shouldContinuePredicting = false
                return endToken
            }
        }
        
        clearBatch()
        addToBatch(token: token, pos: currentTokenCount)
        
        if llama_decode(context, batch) != 0 {
            shouldContinuePredicting = false
            return endToken
        }
        
        currentTokenCount += 1
        return token
    }
    
    func stopGeneration() {
        shouldContinuePredicting = false
    }
    
    func generateResponseStream(from input: String) -> AsyncStream<String> {
        return AsyncStream<String> { continuation in
            Task {
                guard prepareContext(for: input) else { return continuation.finish() }
                while shouldContinuePredicting && currentTokenCount < Int32(maxTokenCount) {
                    let token = predictNextToken()
                    guard token != endToken else { return continuation.finish() }
                    let word = decode(token)
                    continuation.yield(word)
                }
                continuation.finish()
            }
        }
    }
}

public enum LLMError: Error {
    case modelLoadFailed
    case contextCreationFailed
    case tokenizationFailed
    case decodingFailed
    case inputTooLong
    case decodeFailed
}

open class LLM: ObservableObject {
    private(set) var model: Model
    public var history: [Chat]
    public var preprocess: @Sendable (_ input: String, _ history: [Chat]) -> String = { input, _ in return input }
    public var postprocess: @Sendable (_ output: String) -> Void = { print($0) }
    public var update: @Sendable (_ outputDelta: String?) -> Void = { _ in }
    
    @Published public private(set) var output = ""
    
    public var template: Template? = nil {
        didSet {
            guard let template else {
                preprocess = { input, _ in return input }
                Task { await core.setStopSequence(nil) }
                return
            }
            preprocess = template.preprocess
            Task { await core.setStopSequence(template.stopSequence) }
        }
    }
    
    public var seed: UInt32 {
        didSet {
            Task { await core.setParameters(seed: seed) }
        }
    }
    
    public var topK: Int32 {
        didSet {
            Task { await core.setParameters(topK: topK) }
        }
    }
    
    public var topP: Float {
        didSet {
            Task { await core.setParameters(topP: topP) }
        }
    }
    
    public var temp: Float {
        didSet {
            Task { await core.setParameters(temp: temp) }
        }
    }
    
    public var historyLimit: Int
    public var path: [CChar]
    
    public let core: LLMCore
    private var isAvailable = true
    private var input: String = ""
    
    
    public init?(
        from path: String,
        stopSequence: String? = nil,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        historyLimit: Int = 8,
        maxTokenCount: Int32 = 2048
    ) {
        self.path = path.cString(using: .utf8)!
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temp = temp
        self.historyLimit = historyLimit
        self.history = history
        
        var modelParams = llama_model_default_params()
#if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
#endif
        guard let model = llama_model_load_from_file(self.path, modelParams) else {
            return nil
        }
        self.model = model
        
        let finalMaxTokenCount = Int(min(maxTokenCount, llama_model_n_ctx_train(model)))
        
        do {
            self.core = try LLMCore(
                model: model,
                path: self.path,
                seed: seed,
                topK: topK,
                topP: topP,
                temp: temp,
                maxTokenCount: finalMaxTokenCount
            )
            
            if let stopSequence {
                Task {
                    await core.setStopSequence(stopSequence)
                }
            }
        } catch {
            llama_model_free(model)
            return nil
        }
    }
    
    public convenience init?(
        from url: URL,
        stopSequence: String? = nil,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        historyLimit: Int = 8,
        maxTokenCount: Int32 = 2048
    ) {
        self.init(
            from: url.path,
            stopSequence: stopSequence,
            history: history,
            seed: seed,
            topK: topK,
            topP: topP,
            temp: temp,
            historyLimit: historyLimit,
            maxTokenCount: maxTokenCount
        )
    }
    
    public convenience init?(
        from url: URL,
        template: Template,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        historyLimit: Int = 8,
        maxTokenCount: Int32 = 2048
    ) {
        self.init(
            from: url.path,
            stopSequence: template.stopSequence,
            history: history,
            seed: seed,
            topK: topK,
            topP: topP,
            temp: temp,
            historyLimit: historyLimit,
            maxTokenCount: maxTokenCount
        )
        self.preprocess = template.preprocess
        self.template = template
    }
    
    public convenience init?(
        from huggingFaceModel: HuggingFaceModel,
        to url: URL = .documentsDirectory,
        as name: String? = nil,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        historyLimit: Int = 8,
        maxTokenCount: Int32 = 2048,
        updateProgress: @Sendable @escaping (Double) -> Void = { print(String(format: "downloaded(%.2f%%)", $0 * 100)) }
    ) async throws {
        let url = try await huggingFaceModel.download(to: url, as: name) { progress in
            Task { @MainActor in updateProgress(progress) }
        }
        self.init(
            from: url,
            template: huggingFaceModel.template,
            history: history,
            seed: seed,
            topK: topK,
            topP: topP,
            temp: temp,
            historyLimit: historyLimit,
            maxTokenCount: maxTokenCount
        )
    }
    
    deinit {
        llama_model_free(model)
    }
    
    
    @MainActor public func setOutput(to newOutput: consuming String) {
        output = newOutput
    }
    
    public func stop() {
        Task { await core.stopGeneration() }
    }
    
    
    open func recoverFromLengthy(_ input: borrowing String, to output: borrowing AsyncStream<String>.Continuation) {
        output.yield("TL;DR")
    }
    
    public func getCompletion(from input: borrowing String) async -> String {
        guard isAvailable else { return "LLM is being used" }
        
        isAvailable = false
        defer { isAvailable = true }
        
        let response = await core.generateResponseStream(from: input)
        var output = ""
        
        for await responseDelta in response {
            output += responseDelta
        }
        
        return output
    }
    
    public func respond(to input: String, with makeOutputFrom: @escaping (AsyncStream<String>) async -> String) async {
        guard isAvailable else { return }
        
        isAvailable = false
        defer { isAvailable = true }
        
        self.input = input
        let processedInput = preprocess(input, history)
        let response = await core.generateResponseStream(from: processedInput)
        let output = await makeOutputFrom(response)
        
        history += [(.user, input), (.bot, output)]
        let historyCount = history.count
        if historyLimit < historyCount {
            history.removeFirst(min(2, historyCount))
        }
        
        postprocess(output)
    }
    
    open func respond(to input: String) async {
        await respond(to: input) { [weak self] response in
            guard let self = self else { return "" }
            
            await self.setOutput(to: "")
            for await responseDelta in response {
                self.update(responseDelta)
                await self.setOutput(to: self.output + responseDelta)
            }
            self.update(nil)
            
            let trimmedOutput = self.output.trimmingCharacters(in: .whitespacesAndNewlines)
            await self.setOutput(to: trimmedOutput.isEmpty ? "..." : trimmedOutput)
            return self.output
        }
    }
    
    public func encode(_ text: borrowing String, shouldAddBOS: Bool = true) async -> [Token] {
        return await core.encode(text, shouldAddBOS: shouldAddBOS)
    }
}

public enum Role {
    case user
    case bot
}

public struct Template: Sendable {
    public typealias Attachment = (prefix: String, suffix: String)
    public let system: Attachment
    public let user: Attachment
    public let bot: Attachment
    public let systemPrompt: String?
    public let stopSequence: String?
    public let prefix: String
    public let shouldDropLast: Bool
    
    public init(
        prefix: String = "",
        system: Attachment? = nil,
        user: Attachment? = nil,
        bot: Attachment? = nil,
        stopSequence: String? = nil,
        systemPrompt: String?,
        shouldDropLast: Bool = false
    ) {
        self.system = system ?? ("", "")
        self.user = user  ?? ("", "")
        self.bot = bot ?? ("", "")
        self.stopSequence = stopSequence
        self.systemPrompt = systemPrompt
        self.prefix = prefix
        self.shouldDropLast = shouldDropLast
    }
    
    public var preprocess: @Sendable (_ input: String, _ history: [Chat]) -> String {
        return { [self] input, history in
            var processed = prefix
            if let systemPrompt {
                processed += "\(system.prefix)\(systemPrompt)\(system.suffix)"
            }
            for chat in history {
                if chat.role == .user {
                    processed += "\(user.prefix)\(chat.content)\(user.suffix)"
                } else {
                    processed += "\(bot.prefix)\(chat.content)\(bot.suffix)"
                }
            }
            processed += "\(user.prefix)\(input)\(user.suffix)"
            if shouldDropLast {
                processed += bot.prefix.dropLast()
            } else {
                processed += bot.prefix
            }
            return processed
        }
    }
    
    public static func chatML(_ systemPrompt: String? = nil) -> Template {
        return Template(
            system: ("<|im_start|>system\n", "<|im_end|>\n"),
            user: ("<|im_start|>user\n", "<|im_end|>\n"),
            bot: ("<|im_start|>assistant\n", "<|im_end|>\n"),
            stopSequence: "<|im_end|>",
            systemPrompt: systemPrompt
        )
    }
    
    public static func alpaca(_ systemPrompt: String? = nil) -> Template {
        return Template(
            system: ("", "\n\n"),
            user: ("### Instruction:\n", "\n\n"),
            bot: ("### Response:\n", "\n\n"),
            stopSequence: "###",
            systemPrompt: systemPrompt
        )
    }
    
    public static func llama(_ systemPrompt: String? = nil) -> Template {
        return Template(
            prefix: "[INST] ",
            system: ("<<SYS>>\n", "\n<</SYS>>\n\n"),
            user: ("", " [/INST]"),
            bot: (" ", "</s><s>[INST] "),
            stopSequence: "</s>",
            systemPrompt: systemPrompt,
            shouldDropLast: true
        )
    }
    
    public static let mistral = Template(
        user: ("[INST] ", " [/INST]"),
        bot: ("", "</s> "),
        stopSequence: "</s>",
        systemPrompt: nil
    )

    public static let gemma = Template(
        user: ("<start_of_turn>user\n", "<end_of_turn>\n"),
        bot: ("<start_of_turn>model\n", "<end_of_turn>\n"),
        stopSequence: "<end_of_turn>",
        systemPrompt: nil
    )
}

public enum Quantization: String {
    case IQ1_S
    case IQ1_M
    case IQ2_XXS
    case IQ2_XS
    case IQ2_S
    case IQ2_M
    case Q2_K_S
    case Q2_K
    case IQ3_XXS
    case IQ3_XS
    case IQ3_S
    case IQ3_M
    case Q3_K_S
    case Q3_K_M
    case Q3_K_L
    case IQ4_XS
    case IQ4_NL
    case Q4_0
    case Q4_1
    case Q4_K_S
    case Q4_K_M
    case Q5_0
    case Q5_1
    case Q5_K_S
    case Q5_K_M
    case Q6_K
    case Q8_0
}

public enum HuggingFaceError: Error {
    case network(statusCode: Int)
    case noFilteredURL
    case urlIsNilForSomeReason
}

public struct HuggingFaceModel {
    public let name: String
    public let template: Template
    public let filterRegexPattern: String
    
    public init(_ name: String, template: Template, filterRegexPattern: String) {
        self.name = name
        self.template = template
        self.filterRegexPattern = filterRegexPattern
    }
    
    public init(_ name: String, _ quantization: Quantization = .Q4_K_M, template: Template) {
        self.name = name
        self.template = template
        self.filterRegexPattern = "(?i)\(quantization.rawValue)"
    }
    
    package func getDownloadURLStrings() async throws -> [String] {
        let url = URL(string: "https://huggingface.co/\(name)/tree/main")!
        let data = try await url.getData()
        let content = String(data: data, encoding: .utf8)!
        let downloadURLPattern = #"(?<=href=").*\.gguf\?download=true"#
        let matches = try! downloadURLPattern.matches(in: content)
        let root = "https://huggingface.co"
        return matches.map { match in root + match }
    }

    package func getDownloadURL() async throws -> URL? {
        let urlStrings = try await getDownloadURLStrings()
        for urlString in urlStrings {
            let found = try filterRegexPattern.hasMatch(in: urlString)
            if found { return URL(string: urlString)! }
        }
        return nil
    }
    
    public func download(to directory: URL = .documentsDirectory, as name: String? = nil, _ updateProgress: @Sendable @escaping (Double) -> Void) async throws -> URL {
        var destination: URL
        if let name {
            destination = directory.appending(path: name)
            guard !destination.exists else { updateProgress(1); return destination }
        }
        guard let downloadURL = try await getDownloadURL() else { throw HuggingFaceError.noFilteredURL }
        destination = directory.appending(path: downloadURL.lastPathComponent)
        guard !destination.exists else { return destination }
        try await downloadURL.downloadData(to: destination, updateProgress)
        return destination
    }
    
    public static func tinyLLaMA(_ quantization: Quantization = .Q4_K_M, _ systemPrompt: String) -> HuggingFaceModel {
        HuggingFaceModel("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", quantization, template: .chatML(systemPrompt))
    }
}

extension URL {
    @backDeployed(before: iOS 16)
    public func appending(path: String) -> URL {
        appendingPathComponent(path)
    }
    @backDeployed(before: iOS 16)
    public static var documentsDirectory: URL {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    fileprivate var exists: Bool { FileManager.default.fileExists(atPath: path) }
    fileprivate func getData() async throws -> Data {
        let (data, response) = try await URLSession.shared.data(from: self)
        let statusCode = (response as! HTTPURLResponse).statusCode
        guard statusCode / 100 == 2 else { throw HuggingFaceError.network(statusCode: statusCode) }
        return data
    }
    fileprivate func downloadData(to destination: URL, _ updateProgress: @Sendable @escaping (Double) -> Void) async throws {
        var observation: NSKeyValueObservation!
        let url: URL = try await withCheckedThrowingContinuation { continuation in
            let task = URLSession.shared.downloadTask(with: self) { url, response, error in
                if let error { return continuation.resume(throwing: error) }
                guard let url else { return continuation.resume(throwing: HuggingFaceError.urlIsNilForSomeReason) }
                let statusCode = (response as! HTTPURLResponse).statusCode
                guard statusCode / 100 == 2 else { return continuation.resume(throwing: HuggingFaceError.network(statusCode: statusCode)) }
                continuation.resume(returning: url)
            }
            observation = task.progress.observe(\.fractionCompleted) { progress, _ in
                updateProgress(progress.fractionCompleted)
            }
            task.resume()
        }
        _ = observation
        try FileManager.default.moveItem(at: url, to: destination)
    }
}

package extension String {
    func matches(in content: String) throws -> [String] {
        let pattern = try NSRegularExpression(pattern: self)
        let range = NSRange(location: 0, length: content.utf16.count)
        let matches = pattern.matches(in: content, range: range)
        return matches.map { match in String(content[Range(match.range, in: content)!]) }
    }
    func hasMatch(in content: String) throws -> Bool {
        let pattern = try NSRegularExpression(pattern: self)
        let range = NSRange(location: 0, length: content.utf16.count)
        return pattern.firstMatch(in: content, range: range) != nil
    }
    func firstMatch(in content: String) throws -> String? {
        let pattern = try NSRegularExpression(pattern: self)
        let range = NSRange(location: 0, length: content.utf16.count)
        guard let match = pattern.firstMatch(in: content, range: range) else { return nil }
        return String(content[Range(match.range, in: content)!])
    }
}

extension [String] {
    mutating func scoup(_ count: Int) {
        guard 0 < count else { return }
        let firstIndex = count
        let lastIndex = count * 2
        self.removeSubrange(firstIndex..<lastIndex)
    }
}