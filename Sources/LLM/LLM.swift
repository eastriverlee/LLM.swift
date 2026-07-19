import Foundation
import os
import llama
import LlamaChat
@_exported import LLMMacros

public enum ThinkingMode: Sendable {
    case none
    case suppressed
    case enabled
}

/// A token used by the language model.
public typealias Token = llama_token

/// A batch of tokens for processing.
public typealias Batch = llama_batch

/// A pointer to the underlying model.
public typealias Model = OpaquePointer

/// A pointer to the model's vocabulary.
public typealias Vocab = OpaquePointer

/// A chat message consisting of a role and content.
public typealias Chat = (role: Role, content: String)


/// Core actor responsible for thread-safe interactions with the llama.cpp library.
///
/// `LLMCore` handles low-level operations including token encoding/decoding,
/// context management, and inference. It ensures thread safety by using Swift's
/// actor isolation.
public actor LLMCore {
    // the generation loop is one non-suspending actor job, so interruption
    // lives outside actor isolation; it is scoped to a single generation
    // because setting or clearing a shared flag races the next generation's start
    private struct Interruption {
        var currentGeneration: UInt64 = 0
        var interruptedGeneration: UInt64? = nil
    }

    private nonisolated let interruption = OSAllocatedUnfairLock(initialState: Interruption())

    nonisolated func beginGeneration() -> UInt64 {
        interruption.withLock { state in
            state.currentGeneration &+= 1
            return state.currentGeneration
        }
    }

    nonisolated func interrupt(generation: UInt64) {
        interruption.withLock { $0.interruptedGeneration = generation }
    }

    public nonisolated func interrupt() {
        interruption.withLock { $0.interruptedGeneration = $0.currentGeneration }
    }

    public nonisolated func clearInterruption() {
        interruption.withLock { $0.interruptedGeneration = nil }
    }

    nonisolated func isInterrupted(_ generation: UInt64) -> Bool {
        interruption.withLock { $0.interruptedGeneration == generation }
    }

    nonisolated var isInterrupted: Bool {
        interruption.withLock { $0.interruptedGeneration == $0.currentGeneration }
    }

    private let model: Model
    private let vocab: Vocab
    private var context: OpaquePointer
    private var batch: llama_batch
    private let params: llama_context_params
    
    private(set) var seed: UInt32
    private(set) var topK: Int32
    private(set) var topP: Float
    private(set) var temp: Float
    private(set) var repeatPenalty: Float
    private(set) var repetitionLookback: Int32
    
    private let maxTokenCount: Int
    private let totalTokenCount: Int
    
    public lazy var newlineToken: Token = llama_vocab_nl(vocab)
    public lazy var endOfTurnToken: Token = llama_vocab_eot(vocab)
    public lazy var separatorToken: Token = llama_vocab_sep(vocab)
    public lazy var paddingToken: Token = llama_vocab_pad(vocab)
    
    private lazy var startToken: Token = llama_vocab_bos(vocab)
    private lazy var endToken: Token = llama_vocab_eos(vocab)
    private lazy var nullToken: Token = encode("\0", shouldAddBOS: false).first!
    
    private var thinkingStartTokens: [Token]?
    private var thinkingEndTokens: [Token]?
    private var thinkingStartMarker: String?
    private var thinkingEndMarker: String?
    
    private var stopSequenceTokens: [Token]?
    private var tokenBuffer: [Token] = []
    private let tokenDecodeCache = NSCache<NSNumber, NSString>()
    
    private var shouldContinuePredicting = false
    private var currentTokenCount: Int32 = 0
    private var contextTokens: [Token] = []
    private var debugLastGeneratedTokens: [Token] = []

    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private var chatSession: OpaquePointer?
    
    func setParameters(seed: UInt32? = nil, topK: Int32? = nil, topP: Float? = nil, temp: Float? = nil, repeatPenalty: Float? = nil, repetitionLookback: Int32? = nil) {
        if let seed { self.seed = seed }
        if let topK { self.topK = topK }
        if let topP { self.topP = topP }
        if let temp { self.temp = temp }
        if let repeatPenalty { self.repeatPenalty = repeatPenalty }
        if let repetitionLookback { self.repetitionLookback = repetitionLookback }
        
        if seed != nil || topK != nil || topP != nil || temp != nil || repeatPenalty != nil || repetitionLookback != nil {
            recreateSampler()
        }
    }
    
    func setStopSequence(_ sequence: String?) {
        if let sequence {
            stopSequenceTokens = encode(sequence, shouldAddBOS: false)
        } else {
            stopSequenceTokens = nil
        }
    }
    
    func setThinkingTokens(start: [Token]?, end: [Token]?, startMarker: String?, endMarker: String?) {
        thinkingStartTokens = start
        thinkingEndTokens = end
        thinkingStartMarker = startMarker
        thinkingEndMarker = endMarker
    }
    
    private func recreateSampler() {
        if let sampler {
            llama_sampler_free(sampler)
        }
        
        let samplerParams = llama_sampler_chain_default_params()
        sampler = llama_sampler_chain_init(samplerParams)
        
        llama_sampler_chain_add(sampler!, llama_sampler_init_penalties(repetitionLookback, repeatPenalty, 0, 0))
        llama_sampler_chain_add(sampler!, llama_sampler_init_top_k(topK))
        llama_sampler_chain_add(sampler!, llama_sampler_init_top_p(topP, 1))
        llama_sampler_chain_add(sampler!, llama_sampler_init_temp(temp))
        llama_sampler_chain_add(sampler!, llama_sampler_init_dist(seed))
    }
    
    public init(model: Model, path: [CChar], seed: UInt32, topK: Int32, topP: Float, temp: Float, repeatPenalty: Float, repetitionLookback: Int32, maxTokenCount: Int) throws {
        LLM.ensureInitialized()
        self.model = model
        self.vocab = llama_model_get_vocab(model)
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temp = temp
        self.repeatPenalty = repeatPenalty
        self.repetitionLookback = repetitionLookback
        self.maxTokenCount = maxTokenCount
        self.totalTokenCount = Int(llama_vocab_n_tokens(vocab))
        
        var contextParams = llama_context_default_params()
        let processorCount = Int32(ProcessInfo().processorCount)
        contextParams.n_ctx = UInt32(maxTokenCount)
        contextParams.n_batch = contextParams.n_ctx
        contextParams.n_threads = processorCount
        contextParams.n_threads_batch = processorCount
        contextParams.embeddings = true
        self.params = contextParams
        
        guard let context = llama_init_from_model(model, params) else {
            throw LLMError.contextCreationFailed
        }
        self.context = context
        
        self.batch = llama_batch_init(Int32(maxTokenCount), 0, 1)

        self.chatSession = llm_chat_session_create(UnsafeRawPointer(model))

        recreateSampler()
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        if let sampler {
            llama_sampler_free(sampler)
        }
        if let chatSession {
            llm_chat_session_free(chatSession)
        }
        llama_model_free(model)
    }
    
    
    public func encode(_ text: String, shouldAddBOS: Bool = true, special: Bool = true) -> [Token] {
        let count = Int32(text.utf8.count)
        var tokenCount = count + 1
        let cTokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(tokenCount))
        defer { cTokens.deallocate() }
        
        tokenCount = llama_tokenize(vocab, text, count, cTokens, tokenCount, shouldAddBOS, special)
        let tokens = (0..<Int(tokenCount)).map { cTokens[$0] }
        return tokens
    }
    
    public func decode(_ token: Token, special: Bool = false) -> String {
        let cacheKey = special ? token + Int32(totalTokenCount) : token
        if let cached = tokenDecodeCache.object(forKey: NSNumber(value: cacheKey)) {
            return cached as String
        }
        
        var bufferLength = 16
        var buffer: [CChar] = .init(repeating: 0, count: bufferLength)
        var actualLength = Int(llama_token_to_piece(vocab, token, &buffer, Int32(bufferLength), 0, special))
        
        guard actualLength != 0 else { return "" }
        
        if actualLength < 0 {
            bufferLength = -actualLength
            buffer = .init(repeating: 0, count: bufferLength)
            actualLength = Int(llama_token_to_piece(vocab, token, &buffer, Int32(bufferLength), 0, special))
            guard actualLength > 0 else { return "" }
        }
        
        let validBuffer = Array(buffer.prefix(actualLength))
        let bytes = validBuffer.map { UInt8(bitPattern: $0) }
        guard var decoded = String(bytes: bytes, encoding: .utf8) else { return "" }
        
        if decoded.contains("\0") {
            decoded = decoded.filter { $0 != "\0" }
        }
        
        tokenDecodeCache.setObject(decoded as NSString, forKey: NSNumber(value: cacheKey))
        
        return decoded
    }
    
    public func getChatTemplateHint() -> String? {
        guard let template = llama_model_chat_template(model, nil) else { return nil }
        return String(cString: template)
    }
    
    
    func prepareContext(for input: String) -> Bool {
        guard !input.isEmpty else { return false }

        tokenBuffer.removeAll()

        var tokens = encode(input)
        if tokens.last == nullToken { tokens.removeLast() }

        guard !tokens.isEmpty, maxTokenCount > tokens.count else { return false }

        var reusableCount = 0
        while reusableCount < min(tokens.count, contextTokens.count), tokens[reusableCount] == contextTokens[reusableCount] {
            reusableCount += 1
        }
        if reusableCount == tokens.count { reusableCount -= 1 }
        if reusableCount < contextTokens.count {
            llama_memory_seq_rm(llama_get_memory(context), 0, Int32(reusableCount), -1)
            contextTokens.removeSubrange(reusableCount..<contextTokens.count)
        }

        let newTokens = Array(tokens[reusableCount...])
        clearBatch()
        for (i, token) in newTokens.enumerated() {
            addToBatch(token: token, pos: Int32(reusableCount + i), isLogit: i == newTokens.count - 1)
        }
        guard llama_decode(context, batch) == 0 else { return false }

        contextTokens = tokens
        currentTokenCount = Int32(tokens.count)
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
    
    func predictNextToken(excluding: [Token] = []) -> Token {
        guard shouldContinuePredicting, currentTokenCount < Int32(maxTokenCount) else {
            return endToken
        }
        
        guard let sampler = sampler else { return endToken }
        
        let batchIndex = batch.n_tokens - 1
        
        if !excluding.isEmpty, let logits = llama_get_logits_ith(context, batchIndex) {
            for token in excluding {
                logits[Int(token)] = -.infinity
            }
        }
        
        let token = llama_sampler_sample(sampler, context, batchIndex)
        
        if token == endToken {
            return endToken
        }

        tokenBuffer.append(token)
        contextTokens.append(token)
        
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
    
    private func injectTokensIntoContext(_ tokens: [Token]) -> Bool {
        for token in tokens {
            if let sampler {
                llama_sampler_accept(sampler, token)
            }
            clearBatch()
            addToBatch(token: token, pos: currentTokenCount)
            guard llama_decode(context, batch) == 0 else {
                shouldContinuePredicting = false
                return false
            }
            contextTokens.append(token)
            currentTokenCount += 1
        }
        return true
    }

    func resetContext() {
        currentTokenCount = 0
        tokenBuffer.removeAll()
        contextTokens.removeAll()
        shouldContinuePredicting = false
        llama_memory_seq_rm(llama_get_memory(context), -1, -1, -1)
    }
    
    func generateResponseStream(from input: String, thinking: ThinkingMode = .none) -> AsyncStream<String> {
        generateResponseStreamWithThinking(from: input, thinking: thinking).response
    }
    
    func generateThinkingStream(from input: String) -> AsyncStream<String> {
        generateResponseStreamWithThinking(from: input, thinking: .enabled).thinking
    }
    
    func generateResponseStreamWithThinking(from input: String, thinking thinkingMode: ThinkingMode = .none) -> (thinking: AsyncStream<String>, response: AsyncStream<String>) {
        var thinkingContinuation: AsyncStream<String>.Continuation!
        var responseContinuation: AsyncStream<String>.Continuation!
        
        let thinkingStream = AsyncStream<String> { thinkingContinuation = $0 }
        let responseStream = AsyncStream<String> { responseContinuation = $0 }

        let generation = beginGeneration()

        responseContinuation.onTermination = { [weak self] reason in
            if case .cancelled = reason {
                self?.interrupt(generation: generation)
            }
        }

        Task {
            guard prepareContext(for: input) else {
                thinkingContinuation.finish()
                responseContinuation.finish()
                return
            }
            
            if let sampler { llama_sampler_reset(sampler) }
            
            let startMarker = thinkingStartMarker
            let endMarker = thinkingEndMarker
            let maxPendingLength = max(startMarker?.count ?? 0, endMarker?.count ?? 0)
            
            var currentlyInThinkingPhase = thinkingMode == .enabled && startMarker != nil
            var shouldGuaranteeOutput = true
            var pendingText = ""
            
            func stream(_ text: String) {
                guard !text.isEmpty else { return }
                if currentlyInThinkingPhase {
                    thinkingContinuation.yield(text)
                } else {
                    responseContinuation.yield(text)
                }
            }
            
            func streamPendingText() {
                stream(pendingText)
                pendingText = ""
            }
            
            func finishAllStreams() {
                thinkingContinuation.finish()
                responseContinuation.finish()
            }
            
            func transitionToResponsePhase() {
                streamPendingText()
                currentlyInThinkingPhase = false
                shouldGuaranteeOutput = true
                thinkingContinuation.finish()
            }
            
            func foundThinkingStartMarker() -> Bool {
                guard let marker = startMarker, pendingText.hasSuffix(marker) else { return false }
                pendingText.removeLast(marker.count)
                streamPendingText()
                currentlyInThinkingPhase = true
                return true
            }
            
            func foundThinkingEndMarker() -> Bool {
                guard let marker = endMarker, pendingText.hasSuffix(marker) else { return false }
                pendingText.removeLast(marker.count)
                transitionToResponsePhase()
                return true
            }
            
            func streamOldestPendingTextIfNeeded() {
                guard pendingText.count > maxPendingLength else { return }
                let overflowLength = pendingText.count - maxPendingLength
                stream(String(pendingText.prefix(overflowLength)))
                pendingText.removeFirst(overflowLength)
            }
            
            while !isInterrupted(generation) && shouldContinuePredicting && currentTokenCount < Int32(maxTokenCount) {
                let excludedTokens = shouldGuaranteeOutput ? [endToken] : []
                let token = predictNextToken(excluding: excludedTokens)
                shouldGuaranteeOutput = false
                
                let reachedEndOfGeneration = token == endToken
                
                if reachedEndOfGeneration {
                    let stuckInThinkingPhase = currentlyInThinkingPhase && thinkingEndTokens != nil
                    
                    if stuckInThinkingPhase {
                        let injectedSuccessfully = injectTokensIntoContext(thinkingEndTokens! + [newlineToken])
                        guard injectedSuccessfully else {
                            finishAllStreams()
                            return
                        }
                        transitionToResponsePhase()
                        continue
                    }
                    
                    streamPendingText()
                    finishAllStreams()
                    return
                }
                
                pendingText += decode(token)
                
                let detectingThinkingMarkers = thinkingMode != .none
                
                if detectingThinkingMarkers {
                    if foundThinkingStartMarker() { continue }
                    if foundThinkingEndMarker() { continue }
                }
                
                streamOldestPendingTextIfNeeded()
            }
            
            streamPendingText()
            finishAllStreams()
        }
        
        return (thinkingStream, responseStream)
    }
    
    func renderChatPrompt(messagesJSON: String, toolsJSON: String?, enableThinking: Bool) throws -> RenderedPrompt {
        guard let chatSession else { throw LLMError.chatTemplateFailed("chat session unavailable") }
        guard let rendered = llm_chat_render(chatSession, messagesJSON, toolsJSON, true, enableThinking) else {
            throw LLMError.chatTemplateFailed("render returned nothing")
        }
        defer { llm_chat_string_free(rendered) }
        let data = Data(String(cString: rendered).utf8)
        guard let prompt = try? JSONDecoder().decode(RenderedPrompt.self, from: data) else {
            let failure = try? JSONDecoder().decode(WrapperFailure.self, from: data)
            throw LLMError.chatTemplateFailed(failure?.error ?? "unrecognized render output")
        }
        return prompt
    }

    func parseGeneration(_ text: String, isPartial: Bool) -> GeneratedMessage? {
        guard let chatSession, let parsed = llm_chat_parse(chatSession, text, isPartial) else { return nil }
        defer { llm_chat_string_free(parsed) }
        return try? JSONDecoder().decode(GeneratedMessage.self, from: Data(String(cString: parsed).utf8))
    }

    func generateParsedStream(from prompt: String) -> (thinking: AsyncStream<String>, response: AsyncStream<String>, completion: AsyncStream<GeneratedMessage>) {
        var thinkingContinuation: AsyncStream<String>.Continuation!
        var responseContinuation: AsyncStream<String>.Continuation!
        var completionContinuation: AsyncStream<GeneratedMessage>.Continuation!

        let thinkingStream = AsyncStream<String> { thinkingContinuation = $0 }
        let responseStream = AsyncStream<String> { responseContinuation = $0 }
        let completionStream = AsyncStream<GeneratedMessage> { completionContinuation = $0 }

        let generation = beginGeneration()

        responseContinuation.onTermination = { [weak self] reason in
            if case .cancelled = reason {
                self?.interrupt(generation: generation)
            }
        }

        Task {
            defer {
                thinkingContinuation.finish()
                responseContinuation.finish()
                completionContinuation.finish()
            }

            guard prepareContext(for: prompt) else { return }
            if let sampler { llama_sampler_reset(sampler) }

            var rawText = ""
            var streamedReasoning = ""
            var streamedContent = ""

            func streamDeltas(from message: GeneratedMessage) {
                if let reasoning = message.reasoningContent, reasoning.hasPrefix(streamedReasoning), reasoning != streamedReasoning {
                    thinkingContinuation.yield(String(reasoning.dropFirst(streamedReasoning.count)))
                    streamedReasoning = reasoning
                }
                if let content = message.content, content.hasPrefix(streamedContent), content != streamedContent {
                    responseContinuation.yield(String(content.dropFirst(streamedContent.count)))
                    streamedContent = content
                }
            }

            var isFirstToken = true
            while !isInterrupted(generation) && shouldContinuePredicting && currentTokenCount < Int32(maxTokenCount) {
                let token = predictNextToken(excluding: isFirstToken ? [endToken] : [])
                isFirstToken = false
                if token == endToken || token == endOfTurnToken { break }
                rawText += decode(token, special: true)
                if let partial = parseGeneration(rawText, isPartial: true) {
                    streamDeltas(from: partial)
                }
            }

            let message = parseGeneration(rawText, isPartial: false)
                ?? GeneratedMessage(content: rawText, reasoningContent: nil, toolCalls: [])
            streamDeltas(from: message)
            completionContinuation.yield(message)
        }

        return (thinkingStream, responseStream, completionStream)
    }

    func getEmbeddings(from input: String) throws -> [Float] {
        guard !input.isEmpty else { throw LLMError.inputTooLong }
        
        llama_set_embeddings(context, true)
        defer { 
            llama_set_embeddings(context, false)
        }
        
        llama_memory_clear(llama_get_memory(context), false)
        
        let cleanTokens = prepareTokensForEmbeddings(from: input)
        try processBatchForEmbeddings(cleanTokens)
        
        let embeddings = try extractEmbeddingsFromContext()

        llama_memory_clear(llama_get_memory(context), false)
        contextTokens.removeAll()
        currentTokenCount = 0

        return embeddings
    }
    
    private func prepareTokensForEmbeddings(from input: String) -> [Token] {
        var tokens = encode(input)
        if tokens.last == nullToken { tokens.removeLast() }
        return tokens
    }
    
    private func processBatchForEmbeddings(_ tokens: [Token]) throws {
        guard !tokens.isEmpty else { throw LLMError.tokenizationFailed }
        
        clearBatch()
        for (i, token) in tokens.enumerated() {
            addToBatchForEmbeddings(token: token, pos: Int32(i), isLogit: i == tokens.count - 1)
        }
        
        guard llama_decode(context, batch) == 0 else { throw LLMError.embeddingsFailed }
    }
    
    private func addToBatchForEmbeddings(token: Token, pos: Int32, isLogit: Bool = true) {
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
    
    private func extractEmbeddingsFromContext() throws -> [Float] {
        guard let embeddingsPtr = llama_get_embeddings_ith(context, -1) else {
            throw LLMError.embeddingsFailed
        }
        
        let embeddingDimension = Int(llama_model_n_embd(model))
        var embeddingsArray: [Float] = []
        embeddingsArray.reserveCapacity(embeddingDimension)
        
        for i in 0..<embeddingDimension {
            embeddingsArray.append(embeddingsPtr[i])
        }
        
        return embeddingsArray
    }
    
    public func getLastGeneratedTokens() -> [Token] {
        return debugLastGeneratedTokens
    }

    public func getContextTokenCount() -> Int {
        return contextTokens.count
    }
    
    public func generateWithConstraints(from input: String, jsonSchema: String, thinking: ThinkingMode = .suppressed) throws -> String {
        debugLastGeneratedTokens = []
        let generation = beginGeneration()
        guard prepareContext(for: input) else { throw LLMError.contextCreationFailed }

        guard let grammarPointer = llm_chat_grammar_from_json_schema(jsonSchema) else {
            throw LLMError.tokenizationFailed
        }
        defer { llm_chat_string_free(grammarPointer) }
        guard let constrainedSampler = makeConstrainedSampler(grammar: String(cString: grammarPointer)) else {
            throw LLMError.contextCreationFailed
        }
        defer { llama_sampler_free(constrainedSampler) }

        var output = ""
        while !isInterrupted(generation) && shouldContinuePredicting && currentTokenCount < Int32(maxTokenCount) {
            let token = llama_sampler_sample(constrainedSampler, context, batch.n_tokens - 1)
            if token == endToken || token == endOfTurnToken { break }

            clearBatch()
            addToBatch(token: token, pos: currentTokenCount)
            guard llama_decode(context, batch) == 0 else {
                shouldContinuePredicting = false
                throw LLMError.decodingFailed
            }
            contextTokens.append(token)
            currentTokenCount += 1

            output += decode(token)
            debugLastGeneratedTokens.append(token)
        }
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func makeConstrainedSampler(grammar: String) -> UnsafeMutablePointer<llama_sampler>? {
        guard let chain = llama_sampler_chain_init(llama_sampler_chain_default_params()) else { return nil }
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(repetitionLookback, repeatPenalty, 0, 0))
        llama_sampler_chain_add(chain, llama_sampler_init_grammar(vocab, grammar, "root"))
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1))
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
        return chain
    }
}

/// Errors that can occur during LLM operations.
public enum LLMError: Error {
    case modelLoadFailed
    case contextCreationFailed
    case tokenizationFailed
    case decodingFailed
    case inputTooLong
    case decodeFailed
    case embeddingsFailed
    case chatTemplateFailed(String)
}

struct RenderedPrompt: Decodable {
    let prompt: String
    let additionalStops: [String]
}

struct WrapperFailure: Decodable {
    let error: String
}

struct ChatToolCall: Codable {
    struct Function: Codable {
        let name: String
        let arguments: String
    }

    let type: String
    let function: Function
    let id: String?

    init(name: String, arguments: String, id: String? = nil) {
        self.type = "function"
        self.function = Function(name: name, arguments: arguments)
        self.id = id
    }
}

struct ChatMessage: Codable {
    let role: String
    let content: String
    var toolCalls: [ChatToolCall]? = nil
    var toolCallID: String? = nil

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

struct GeneratedMessage: Decodable {
    let content: String?
    let reasoningContent: String?
    let toolCalls: [ChatToolCall]

    enum CodingKeys: String, CodingKey {
        case content
        case reasoningContent = "reasoning_content"
        case toolCalls = "tool_calls"
    }

    init(content: String?, reasoningContent: String?, toolCalls: [ChatToolCall]) {
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        content = try? container.decode(String.self, forKey: .content)
        reasoningContent = try? container.decode(String.self, forKey: .reasoningContent)
        toolCalls = (try? container.decode([ChatToolCall].self, forKey: .toolCalls)) ?? []
    }
}

/// A function the model can call by generating its name and arguments.
///
/// Declare arguments as a nested `@Generatable` type; its JSON schema is
/// advertised to the model and generated arguments are decoded into it.
///
/// ```swift
/// struct GetWeather: Tool {
///     let description = "Get the current weather for a city"
///
///     @Generatable
///     struct Arguments {
///         let city: String
///     }
///
///     func call(_ arguments: Arguments) async throws -> String {
///         try await weatherAPI.fetch(arguments.city)
///     }
/// }
/// ```
public protocol Tool: Sendable {
    associatedtype Arguments: Generatable
    var name: String { get }
    var description: String { get }
    func call(_ arguments: Arguments) async throws -> String
}

public extension Tool {
    var name: String { String(describing: Self.self) }
}

extension Tool {
    var argumentsSchema: String { Arguments.jsonSchema }

    func invoke(_ argumentsJSON: String) async throws -> String {
        guard let arguments = try? JSONDecoder().decode(Arguments.self, from: Data(argumentsJSON.utf8)) else {
            throw ToolError.argumentDecodingFailed(argumentsJSON)
        }
        return try await call(arguments)
    }
}

/// Errors that can occur while invoking a tool.
public enum ToolError: Error {
    case argumentDecodingFailed(String)
}

/// A record of one tool invocation the model made during a response.
public struct ToolCall: Sendable, Equatable {
    public let name: String
    public let arguments: String
    public let result: String
}

extension [any Tool] {
    var signaturesJSON: String? {
        guard !isEmpty else { return nil }
        let signatures = compactMap { tool -> [String: Any]? in
            guard let parameters = try? JSONSerialization.jsonObject(with: Data(tool.argumentsSchema.utf8)) else { return nil }
            return ["type": "function", "function": ["name": tool.name, "description": tool.description, "parameters": parameters]]
        }
        guard let data = try? JSONSerialization.data(withJSONObject: signatures) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

/// A container for text embeddings generated by the model.
public struct Embeddings: Equatable {
    public let values: [Float]
    public let dimension: Int
    
    public init(values: [Float]) {
        self.values = values
        self.dimension = values.count
    }
    
    public func compare(with other: Embeddings) -> Double {
        guard dimension == other.dimension else { return 0.0 }
        
        let dotProduct = zip(values, other.values).reduce(0) { result, pair in result + pair.0 * pair.1 }
        let magnitudeA = sqrt(values.reduce(0) { $0 + $1 * $1 })
        let magnitudeB = sqrt(other.values.reduce(0) { $0 + $1 * $1 })
        
        guard magnitudeA > 0 && magnitudeB > 0 else { return 0.0 }
        
        let similarity = dotProduct / (magnitudeA * magnitudeB)
        return max(0, min(1, Double(similarity)))
    }
    
    public func findMostSimilar(in candidates: Embeddings...) -> Embeddings {
        guard !candidates.isEmpty else { return self }
        
        let similarities = candidates.map { compare(with: $0) }
        let maxIndex = similarities.enumerated().max { $0.element < $1.element }?.offset ?? 0
        
        return candidates[maxIndex]
    }
}

/// A wrapper for structured output generated according to a JSON schema.
///
/// This type ensures that the model's output conforms to the specified
/// structure defined by the `Generatable` protocol.
public struct StructuredOutput<T: Generatable> {
    public let value: T
    public let rawOutput: String
    
    public init(value: T, rawOutput: String) {
        self.value = value
        self.rawOutput = rawOutput
    }
}

/// Errors that can occur during structured output generation.
public enum StructuredOutputError: Error {
    case invalidJSON
    case schemaMismatch
    case decodingFailed
}

/// Main interface for interacting with Large Language Models in Swift applications.
///
/// `LLM` provides a high-level, SwiftUI-compatible interface for running language models
/// locally using llama.cpp. It manages conversation history, provides async/await methods
/// for generation, and integrates seamlessly with SwiftUI through `ObservableObject`.
///
/// ## Usage
///
/// ```swift
/// let llm = try await LLM(from: .bundle(resource: "model", extension: "gguf"))
/// let response = try await llm.generate(from: "Hello, how are you?")
/// ```
///
/// ## Features
///
/// - **Conversation Management**: Automatically maintains chat history
/// - **SwiftUI Integration**: Works as an `ObservableObject` for reactive UI updates
/// - **Async/Await Support**: Modern Swift concurrency for non-blocking operations
/// - **Streaming Responses**: Real-time token generation via `AsyncStream`
/// - **Template Support**: Built-in chat templates for popular model formats
/// - **Structured Output**: Generate JSON conforming to specified schemas
open class LLM: ObservableObject {
    private(set) var model: Model
    public var history: [Chat]
    public var preprocess: @Sendable (_ input: String, _ history: [Chat], _ thinking: ThinkingMode) -> String = { input, _, _ in return input }
    public var postprocess: @Sendable (_ output: String) -> Void = { print($0) }
    public var update: @Sendable (_ outputDelta: String?) -> Void = { _ in }
    public var updateThinking: @Sendable (_ thinkingDelta: String?) -> Void = { _ in }
    
    @Published public private(set) var output = ""
    @Published public private(set) var thinking = ""
    @Published public private(set) var toolCalls: [ToolCall] = []

    public var systemPrompt: String? = nil
    public var tools: [any Tool] = []
    public var maxToolTurns = 4
    
    public var template: Template? = nil {
        didSet {
            guard let template else {
                preprocess = { input, _, _ in return input }
                Task {
                    await core.setStopSequence(nil)
                    await core.setThinkingTokens(start: nil, end: nil, startMarker: nil, endMarker: nil)
                }
                return
            }
            preprocess = template.preprocess
            Task {
                await core.setStopSequence(template.stopSequence)
                await setupThinkingTokens(from: template)
            }
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

    public var repeatPenalty: Float {
        didSet {
            Task { await core.setParameters(repeatPenalty: repeatPenalty) }
        }
    }
    
    public var repetitionLookback: Int32 {
        didSet {
            Task { await core.setParameters(repetitionLookback: repetitionLookback) }
        }
    }
    
    public var historyLimit: Int
    public var path: [CChar]
    
    public let core: LLMCore
    
    private var isAvailable = true
    private var input: String = ""
    
    static var isLogSilenced = false
    
    fileprivate static func ensureInitialized() {
        struct Initialization {
            static let invoke: Void = {
                llama_backend_init()
            }()
        }
        _ = Initialization.invoke
    }

    static func silenceLogging() {
        guard !isLogSilenced else { return }
        isLogSilenced = true
        let noopCallback: @convention(c) (ggml_log_level, UnsafePointer<CChar>?, UnsafeMutableRawPointer?) -> Void = { _, _, _ in }
        llama_log_set(noopCallback, nil)
        ggml_log_set(noopCallback, nil)
    }
    
    
    public init?(
        from path: String,
        stopSequence: String? = nil,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        repeatPenalty: Float = 1.2,
        repetitionLookback: Int32 = 64,
        historyLimit: Int = 8,
        maxTokenCount: Int32 = 2048
    ) {
        LLM.silenceLogging()
        self.path = path.cString(using: .utf8)!
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.temp = temp
        self.historyLimit = historyLimit
        self.history = history
        self.repeatPenalty = repeatPenalty
        self.repetitionLookback = repetitionLookback
        
        #if DEBUG
        print("GNERATING WITH SEEED: \(seed)")
        #endif
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
                repeatPenalty: repeatPenalty,
                repetitionLookback: repetitionLookback,
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
        repeatPenalty: Float = 1.2,
        repetitionLookback: Int32 = 64,
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
            repeatPenalty: repeatPenalty,
            repetitionLookback: repetitionLookback,
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
        repeatPenalty: Float = 1.2,
        repetitionLookback: Int32 = 64,
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
            repeatPenalty: repeatPenalty,
            repetitionLookback: repetitionLookback,
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
        repeatPenalty: Float = 1.2,
        repetitionLookback: Int32 = 64,
        historyLimit: Int = 8,
        maxTokenCount: Int32 = 2048,
        updateProgress: @Sendable @escaping (Double) -> Void = { print(String(format: "downloaded(%.2f%%)", $0 * 100)) }
    ) async throws {
        let url = try await huggingFaceModel.download(to: url, as: name) { progress in
            Task { @MainActor in updateProgress(progress) }
        }
        if let template = huggingFaceModel.template {
            self.init(
                from: url,
                template: template,
                history: history,
                seed: seed,
                topK: topK,
                topP: topP,
                temp: temp,
                repeatPenalty: repeatPenalty,
                repetitionLookback: repetitionLookback,
                historyLimit: historyLimit,
                maxTokenCount: maxTokenCount
            )
            await setupThinkingTokens(from: template)
        } else {
            self.init(
                from: url,
                history: history,
                seed: seed,
                topK: topK,
                topP: topP,
                temp: temp,
                repeatPenalty: repeatPenalty,
                repetitionLookback: repetitionLookback,
                historyLimit: historyLimit,
                maxTokenCount: maxTokenCount
            )
        }
    }
    
    private func setupThinkingTokens(from template: Template?) async {
        guard let template else { return }
        let (startTokens, startMarker) = await convertToTokensAndMarker(template.thinkingStart)
        let (endTokens, endMarker) = await convertToTokensAndMarker(template.thinkingEnd)
        await core.setThinkingTokens(start: startTokens, end: endTokens, startMarker: startMarker, endMarker: endMarker)
    }
    
    private func convertToTokensAndMarker(_ sequence: TokenSequence?) async -> ([Token]?, String?) {
        guard let sequence else { return (nil, nil) }
        switch sequence {
        case .string(let text):
            return (await core.encode(text, shouldAddBOS: false, special: false), text)
        case .token(let token):
            return ([token], await core.decode(token, special: true))
        }
    }
    
    @MainActor public func setOutput(to newOutput: consuming String) {
        output = newOutput
    }
    
    @MainActor public func setThinking(to newThinking: consuming String) {
        thinking = newThinking
    }
    
    public func stop() {
        core.interrupt()
    }
    
    public func reset() {
        history.removeAll()
        Task { await core.resetContext() }
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
        
        for await content in response {
            output += content
        }
        
        return output
    }
    
    public func respond(to input: String, thinking: ThinkingMode = .none, with makeOutputFrom: @escaping (AsyncStream<String>) async -> String) async {
        guard isAvailable else { return }
        
        isAvailable = false
        defer { isAvailable = true }
        
        self.input = input
        let processedInput = preprocess(input, history, thinking)
        
        let response = await core.generateResponseStream(from: processedInput, thinking: thinking)
        let output = await makeOutputFrom(response)
        
        history += [(.user, input), (.bot, output)]
        let historyCount = history.count
        if historyLimit < historyCount {
            history.removeFirst(min(2, historyCount))
        }
        
        postprocess(output)
    }
    
    open func respond(to input: String, thinking: ThinkingMode = .none) async {
        guard isAvailable else { return }

        isAvailable = false
        defer { isAvailable = true }

        self.input = input

        await setOutput(to: "")
        await setThinking(to: "")

        if template == nil {
            await respondUsingChatTemplate(to: input, thinking: thinking)
        } else {
            await respondUsingManualTemplate(to: input, thinking: thinking)
        }

        let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
        await setOutput(to: trimmedOutput.isEmpty ? "..." : trimmedOutput)

        history += [(.user, input), (.bot, output)]
        let historyCount = history.count
        if historyLimit < historyCount {
            history.removeFirst(min(2, historyCount))
        }

        postprocess(output)
    }

    private func respondUsingManualTemplate(to input: String, thinking: ThinkingMode) async {
        let processedInput = preprocess(input, history, thinking)
        let (thinkingStream, responseStream) = await core.generateResponseStreamWithThinking(from: processedInput, thinking: thinking)
        await consume(thinkingStream: thinkingStream, responseStream: responseStream)
    }

    private func respondUsingChatTemplate(to input: String, thinking: ThinkingMode) async {
        await clearToolCalls()
        var messages = composeMessages(endingWith: input)

        for _ in 0..<Swift.max(1, maxToolTurns) {
            guard let prompt = await renderPrompt(for: messages, thinking: thinking) else {
                await respondUsingManualTemplate(to: input, thinking: thinking)
                return
            }

            let (thinkingStream, responseStream, completionStream) = await core.generateParsedStream(from: prompt)
            await consume(thinkingStream: thinkingStream, responseStream: responseStream)

            var generated: GeneratedMessage? = nil
            for await message in completionStream { generated = message }
            guard let message = generated, !message.toolCalls.isEmpty else { return }

            messages.append(ChatMessage(role: "assistant", content: message.content ?? "", toolCalls: message.toolCalls))
            for call in message.toolCalls {
                let result = await invokeTool(named: call.function.name, argumentsJSON: call.function.arguments)
                await recordToolCall(ToolCall(name: call.function.name, arguments: call.function.arguments, result: result))
                messages.append(ChatMessage(role: "tool", content: result, toolCallID: call.id))
            }

            if core.isInterrupted { return }
        }
    }

    private func consume(thinkingStream: AsyncStream<String>, responseStream: AsyncStream<String>) async {
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                for await content in thinkingStream {
                    self.updateThinking(content)
                    await self.setThinking(to: self.thinking + content)
                }
                self.updateThinking(nil)
            }

            group.addTask {
                for await content in responseStream {
                    self.update(content)
                    await self.setOutput(to: self.output + content)
                }
                self.update(nil)
            }
        }
    }

    private func composeMessages(endingWith input: String) -> [ChatMessage] {
        var messages: [ChatMessage] = []
        if let systemPrompt {
            messages.append(ChatMessage(role: "system", content: systemPrompt))
        }
        for chat in history {
            messages.append(ChatMessage(role: chat.role == .user ? "user" : "assistant", content: chat.content))
        }
        messages.append(ChatMessage(role: "user", content: input))
        return messages
    }

    private func renderPrompt(for messages: [ChatMessage], thinking: ThinkingMode) async -> String? {
        guard let messagesData = try? JSONEncoder().encode(messages),
              let messagesJSON = String(data: messagesData, encoding: .utf8) else { return nil }
        let rendered = try? await core.renderChatPrompt(
            messagesJSON: messagesJSON,
            toolsJSON: tools.signaturesJSON,
            enableThinking: thinking != .suppressed
        )
        guard let rendered else { return nil }
        await core.setStopSequence(rendered.additionalStops.first)
        return rendered.prompt
    }

    private func invokeTool(named name: String, argumentsJSON: String) async -> String {
        guard let tool = tools.first(where: { $0.name == name }) else {
            return "unknown tool: \(name)"
        }
        do {
            return try await tool.invoke(argumentsJSON)
        } catch {
            return "tool failed: \(error)"
        }
    }

    @MainActor private func recordToolCall(_ toolCall: ToolCall) {
        toolCalls.append(toolCall)
    }

    @MainActor private func clearToolCalls() {
        toolCalls.removeAll()
    }
    
    public func encode(_ text: borrowing String, shouldAddBOS: Bool = true) async -> [Token] {
        return await core.encode(text, shouldAddBOS: shouldAddBOS)
    }
    
    public func getEmbeddings(_ text: String) async throws -> Embeddings {
        let values = try await core.getEmbeddings(from: text)
        return Embeddings(values: values)
    }
    
    public func respond<T: Generatable>(
        to prompt: String,
        as type: T.Type,
        thinking: ThinkingMode = .none
    ) async throws -> StructuredOutput<T> {
        let schemaPrompt = """
        \(prompt)
        
        Generate a JSON response that:
        Matches this exact schema: \(T.jsonSchema)
        Return only the JSON object, no other text.
        """
        let processedPrompt = preprocess(schemaPrompt, history, thinking)
        
        let rawOutput = try await core.generateWithConstraints(
            from: processedPrompt,
            jsonSchema: T.jsonSchema,
            thinking: thinking
        )
        
        guard let jsonData = rawOutput.data(using: .utf8) else {
            throw StructuredOutputError.invalidJSON
        }
        
        do {
            let decodedValue = try JSONDecoder().decode(T.self, from: jsonData)
            let output = StructuredOutput(value: decodedValue, rawOutput: rawOutput)
            history += [(.user, processedPrompt), (.bot, rawOutput)]
            let historyCount = history.count
            if historyLimit < historyCount {
                history.removeFirst(min(2, historyCount))
            }
            postprocess(rawOutput)
            return output
        } catch {
            print("JSON Decoding failed:")
            print("Raw output: '\(rawOutput)'")
            print("JSON data: '\(String(data: jsonData, encoding: .utf8) ?? "nil")'")
            print("Error: \(error)")
            print("Schema: \(T.jsonSchema)")
            throw StructuredOutputError.decodingFailed
        }
    }
}

/// The role of a participant in a chat conversation.
public enum Role {
    case user
    case bot
}

/// A sequence of tokens that can be represented as either a string or a single token ID.
public enum TokenSequence: Sendable, Equatable {
    case string(String)
    case token(Token)
    
    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }
}

extension TokenSequence: ExpressibleByStringLiteral {
    public init(stringLiteral value: String) {
        self = .string(value)
    }
}

extension TokenSequence: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: Int32) {
        self = .token(value)
    }
}

/// A chat template for formatting conversations according to model-specific formats.
///
/// Templates handle the preprocessing of user input and conversation history
/// into formats expected by different language models, including system prompts,
/// message formatting, and stop sequences.
public struct Template: Sendable {
    public typealias Attachment = (prefix: String, suffix: String)
    public let system: Attachment
    public let user: Attachment
    public let bot: Attachment
    public let systemPrompt: String?
    public let stopSequence: String?
    public let thinkingStart: TokenSequence?
    public let thinkingEnd: TokenSequence?
    public let prefix: String
    public let shouldDropLast: Bool
    
    public init(
        prefix: String = "",
        system: Attachment? = nil,
        user: Attachment? = nil,
        bot: Attachment? = nil,
        stopSequence: String? = nil,
        thinkingStart: TokenSequence? = nil,
        thinkingEnd: TokenSequence? = nil,
        systemPrompt: String?,
        shouldDropLast: Bool = false
    ) {
        self.system = system ?? ("", "")
        self.user = user  ?? ("", "")
        self.bot = bot ?? ("", "")
        self.stopSequence = stopSequence
        self.thinkingStart = thinkingStart
        self.thinkingEnd = thinkingEnd
        self.systemPrompt = systemPrompt
        self.prefix = prefix
        self.shouldDropLast = shouldDropLast
    }
    
    public var preprocess: @Sendable (_ input: String, _ history: [Chat], _ thinking: ThinkingMode) -> String {
        return { [self] input, history, thinking in
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
            
            if thinking == .enabled, let start = thinkingStart?.stringValue {
                processed += start
            } else if thinking == .suppressed, let start = thinkingStart?.stringValue, let end = thinkingEnd?.stringValue {
                processed += start + end
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
            thinkingStart: "<think>",
            thinkingEnd: "</think>",
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
            thinkingStart: "[THINK]",
            thinkingEnd: "[/THINK]",
            systemPrompt: systemPrompt,
            shouldDropLast: true
        )
    }
    
    public static let mistral = Template(
        user: ("[INST] ", " [/INST]"),
        bot: ("", "</s> "),
        stopSequence: "</s>",
        thinkingStart: "[THINK]",
        thinkingEnd: "[/THINK]",
        systemPrompt: nil
    )
    
    public static let gemma = Template(
        user: ("<start_of_turn>user\n", "<end_of_turn>\n"),
        bot: ("<start_of_turn>model\n", "<end_of_turn>\n"),
        stopSequence: "<end_of_turn>",
        thinkingStart: "<think>",
        thinkingEnd: "</think>",
        systemPrompt: nil
    )
}

/// Quantization levels available for model compression.
///
/// Quantization reduces model size and memory usage at the cost of some accuracy.
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

/// Errors that can occur when downloading models from Hugging Face.
public enum HuggingFaceError: Error {
    case network(statusCode: Int)
    case noFilteredURL
    case urlIsNilForSomeReason
}

/// A utility for downloading and managing models from Hugging Face.
///
/// This struct provides methods to download GGUF models directly from
/// Hugging Face repositories with support for different quantization levels.
public struct HuggingFaceModel {
    public let name: String
    public let template: Template?
    public let filterRegexPattern: String

    public init(_ name: String, template: Template? = nil, filterRegexPattern: String) {
        self.name = name
        self.template = template
        self.filterRegexPattern = filterRegexPattern
    }

    public init(_ name: String, _ quantization: Quantization? = .Q4_K_M, template: Template? = nil) {
        self.name = name
        self.template = template
        self.filterRegexPattern = quantization.map { "(?i)\($0.rawValue)" } ?? ".*"
    }
    
    package func getDownloadURLStrings() async throws -> [String] {
        let url = URL(string: "https://huggingface.co/\(name)/tree/main")!
        let data = try await url.getData()
        let content = String(data: data, encoding: .utf8)!
        let downloadURLPattern = #"(?<=href=")[^"]*\.gguf\?download=true"#
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

extension Array where Element: Equatable {
    var hasThreeSameElementsAtTheEnd: Bool {
        count >= 3 && self[count - 3] == self[count - 2] && self[count - 2] == self[count - 1]
    }
}

extension Character {
    var isValidStringCharacter: Bool {
        guard self != "\"" && self != "\\" else { return false }
        return isLetter || self == " " || isNumber || isLowercase || isUppercase || isASCII && isPunctuation || isASCII && isSymbol
    }
}
