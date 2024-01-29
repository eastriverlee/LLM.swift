import Foundation
import llama

public typealias Token = llama_token
public typealias Model = OpaquePointer
public typealias Chat = (role: Role, content: String)

@globalActor public actor InferenceActor {
    static public let shared = InferenceActor()
}

open class LLM: ObservableObject {
    public var model: Model
    public var history: [Chat]
    public var preProcess: (_ input: String, _ history: [Chat]) -> String = { input, _ in return input }
    public var postProcess: (_ output: String) -> Void                    = { print($0) }
    public var update: (_ outputDelta: String?) -> Void                   = { _ in }
    public var template: Template? = nil {
        didSet {
            guard let template else {
                preProcess = { input, _ in return input }
                stopSequence = nil
                stopSequenceLength = 0
                return
            }
            preProcess = template.preProcess
            if let stopSequence = template.stopSequence?.utf8CString {
                self.stopSequence = stopSequence
                stopSequenceLength = stopSequence.count - 1
            } else {
                stopSequence = nil
                stopSequenceLength = 0
            }
        }
    }
    
    public var topK: Int32
    public var topP: Float
    public var temp: Float
    public var historyLimit: Int
    public var path: [CChar]
    
    @Published public private(set) var output = ""    
    @MainActor public func setOutput(to newOutput: consuming String) {
        output = newOutput
    }
    
    private var context: Context!
    private var batch: llama_batch!
    private let maxTokenCount: Int
    private let totalTokenCount: Int
    private let newlineToken: Token
    private var stopSequence: ContiguousArray<CChar>?
    private var stopSequenceLength: Int
    private var params: llama_context_params
    private var isFull = false
    
    public init(
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
        var modelParams = llama_model_default_params()
#if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
#endif
        let model = llama_load_model_from_file(self.path, modelParams)!
        params = llama_context_default_params()
        let processorCount = UInt32(ProcessInfo().processorCount)
        self.maxTokenCount = Int(min(maxTokenCount, llama_n_ctx_train(model)))
        params.seed = seed
        params.n_ctx = UInt32(maxTokenCount) + (maxTokenCount % 2 == 1 ? 1 : 2)
        params.n_batch = params.n_ctx
        params.n_threads = processorCount
        params.n_threads_batch = processorCount
        self.topK = topK
        self.topP = topP
        self.temp = temp
        self.historyLimit = historyLimit
        self.model = model
        self.history = history
        self.totalTokenCount = Int(llama_n_vocab(model))
        self.newlineToken = llama_token_nl(model)
        self.stopSequence = stopSequence?.utf8CString
        self.stopSequenceLength = (self.stopSequence?.count ?? 1) - 1
        batch = llama_batch_init(Int32(self.maxTokenCount), 0, 1)
    }
    
    deinit {
        llama_free_model(model)
    }
    
    public convenience init(
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
    
    public convenience init(
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
            history: history,
            seed: seed,
            topK: topK,
            topP: topP,
            temp: temp,
            historyLimit: historyLimit,
            maxTokenCount: maxTokenCount
        )
        self.template = template
    }
    
    private var shouldContinuePredicting = false
    public func stop() {
        shouldContinuePredicting = false
    }
    
    @InferenceActor
    private func predictNextToken() async -> Token {
        guard shouldContinuePredicting else { return llama_token_eos(model) }
        let logits = llama_get_logits_ith(context.pointer, batch.n_tokens - 1)!
        var candidates: [llama_token_data] = (0..<totalTokenCount).map { token in
            llama_token_data(id: Int32(token), logit: logits[token], p: 0.0)
        }
        var token: llama_token!
        candidates.withUnsafeMutableBufferPointer { pointer in
            var candidates = llama_token_data_array(
                data: pointer.baseAddress,
                size: totalTokenCount,
                sorted: false
            )
            llama_sample_top_k(context.pointer, &candidates, topK, 1)
            llama_sample_top_p(context.pointer, &candidates, topP, 1)
            llama_sample_temp(context.pointer, &candidates, temp)
            token = llama_sample_token(context.pointer, &candidates)
        }
        batch.clear()
        batch.add(token, currentCount, [0], true)
        context.decode(batch)
        return token
    }
    
    private var currentCount: Int32!
    private var decoded = ""
    
    private func prepare(from input: borrowing String, to output: borrowing AsyncStream<String>.Continuation) -> Bool {
        guard !input.isEmpty else { return false }
        context = .init(model, params)
        var tokens = encode(input)
        var initialCount = tokens.count
        currentCount = Int32(initialCount)
        if maxTokenCount <= currentCount {
            if history.isEmpty {
                isFull = true
                output.yield("Input is too long.")
                return false
            } else {
                history.removeFirst(min(2, history.count))
                tokens = encode(preProcess(self.input, history))
                initialCount = tokens.count
                currentCount = Int32(initialCount)
            }
        }
        for (i, token) in tokens.enumerated() {
            batch.n_tokens = Int32(i)
            batch.add(token, batch.n_tokens, [0], i == initialCount - 1)
        }
        context.decode(batch)
        shouldContinuePredicting = true
        return true
    }
    
    @InferenceActor
    private func finishResponse(from response: inout [String], to output: borrowing AsyncStream<String>.Continuation) async {
        multibyteCharacter.removeAll()
        var input = ""
        if !history.isEmpty {
            history.removeFirst(min(2, history.count))
            input = preProcess(self.input, history)
        } else {
            response.scoup(response.count / 3)
            input = preProcess(self.input, history)
            input += response.joined()
        }
        let rest = getResponse(from: input)
        for await restDelta in rest {
            output.yield(restDelta)
        }
    }
    
    private func process(_ token: Token, to output: borrowing AsyncStream<String>.Continuation) -> Bool {
        struct saved {
            static var endIndex = 0
            static var letters: [CChar] = []
        }
        guard token != llama_token_eos(model) else { return false }
        var word = decode(token)
        guard let stopSequence else { output.yield(word); return true }
        var found = 0 < saved.endIndex
        var letters: [CChar] = []
        for letter in word.utf8CString {
            guard letter != 0 else { break }
            if letter == stopSequence[saved.endIndex] {
                saved.endIndex += 1
                found = true
                saved.letters.append(letter)
                guard saved.endIndex == stopSequenceLength else { continue }
                saved.endIndex = 0
                saved.letters.removeAll()
                return false
            } else if found {
                saved.endIndex = 0
                if !saved.letters.isEmpty {
                    word = String(cString: saved.letters + [0]) + word
                    saved.letters.removeAll()
                }
                output.yield(word)
                return true
            }
            letters.append(letter)
        }
        if !letters.isEmpty { output.yield(found ? String(cString: letters + [0]) : word) }
        return true
    }
    
    private func getResponse(from input: borrowing String) -> AsyncStream<String> {
        .init { output in Task {
            defer { context = nil }
            guard prepare(from: input, to: output) else { return output.finish() }
            var response: [String] = []
            while currentCount < maxTokenCount {
                let token = await predictNextToken()
                if !process(token, to: output) { return output.finish() }
                currentCount += 1
            }
            await finishResponse(from: &response, to: output)
            return output.finish()
        } }
    }
    
    private var input: String = ""
    private var isAvailable = true
    
    @InferenceActor
    public func getCompletion(from input: borrowing String) async -> String {
        guard isAvailable else { fatalError("LLM is being used") }
        isAvailable = false
        let response = getResponse(from: input)
        var output = ""
        for await responseDelta in response {
            output += responseDelta
        }
        isAvailable = true
        return output
    }
    
    @InferenceActor
    public func respond(to input: String, with makeOutputFrom: @escaping (AsyncStream<String>) async -> String) async {
        guard isAvailable else { return }
        isAvailable = false
        self.input = input
        let processedInput = preProcess(input, history)
        let response = getResponse(from: processedInput)
        let output = await makeOutputFrom(response)
        history += [(.user, input), (.bot, output)]
        let historyCount = history.count
        if historyLimit < historyCount {
            history.removeFirst(min(2, historyCount))
        }
        postProcess(output)
        isAvailable = true
    }
    
    open func respond(to input: String) async {
        await respond(to: input) { [self] response in
            await setOutput(to: "")
            for await responseDelta in response {
                update(responseDelta)
                await setOutput(to: output + responseDelta)
            }
            update(nil)
            let trimmedOutput = output.trimmingCharacters(in: .whitespacesAndNewlines)
            await setOutput(to: trimmedOutput.isEmpty ? "..." : trimmedOutput)
            return output
        }
    }

    private var multibyteCharacter: [CUnsignedChar] = []
    public func decode(_ token: Token) -> String {
        return model.decode(token, with: &multibyteCharacter)
    }
    
    @inlinable
    public func encode(_ text: borrowing String) -> [Token] {
        model.encode(text)
    }
}

extension Model {
    public func shouldAddBOS() -> Bool {
        let addBOS = llama_add_bos_token(self);
        guard addBOS != -1 else {
            return llama_vocab_type(self) == LLAMA_VOCAB_TYPE_SPM
        }
        return addBOS != 0
    }
    
    public func decodeOnly(_ token: Token) -> String {
        var nothing: [CUnsignedChar] = []
        return decode(token, with: &nothing)
    }
    
    public func decode(_ token: Token, with multibyteCharacter: inout [CUnsignedChar]) -> String {
        var bufferLength = 16
        var buffer: [CChar] = .init(repeating: 0, count: bufferLength)
        let actualLength = Int(llama_token_to_piece(self, token, &buffer, Int32(bufferLength)))
        guard 0 != actualLength else { return "" }
        if actualLength < 0 {
            bufferLength = -actualLength
            buffer = .init(repeating: 0, count: bufferLength)
            llama_token_to_piece(self, token, &buffer, Int32(bufferLength))
        } else {
            buffer.removeLast(bufferLength - actualLength)
        }
        if multibyteCharacter.isEmpty, let decoded = String(cString: buffer + [0], encoding: .utf8) {
            return decoded
        }
        multibyteCharacter.append(contentsOf: buffer.map { CUnsignedChar(bitPattern: $0) })
        guard let decoded = String(data: .init(multibyteCharacter), encoding: .utf8) else { return "" }
        multibyteCharacter.removeAll(keepingCapacity: true)
        return decoded
    }

    public func encode(_ text: borrowing String) -> [Token] {
        let addBOS = true
        let count = Int32(text.cString(using: .utf8)!.count)
        var tokenCount = count + 1
        let cTokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(tokenCount)); defer { cTokens.deallocate() }
        tokenCount = llama_tokenize(self, text, count, cTokens, tokenCount, addBOS, false)
        let tokens = (0..<Int(tokenCount)).map { cTokens[$0] }
        return tokens
    }
}

private class Context {
    let pointer: OpaquePointer
    init(_ model: Model, _ params: llama_context_params) {
        self.pointer = llama_new_context_with_model(model, params)
    }
    deinit {
        llama_free(pointer)
    }
    func decode(_ batch: llama_batch) {
        guard llama_decode(pointer, batch) == 0 else { fatalError("llama_decode failed") }
    }
}

extension llama_batch {
    mutating func clear() {
        self.n_tokens = 0
    }
    
    mutating func add(_ token: Token, _ position: Int32, _ ids: [Int], _ logit: Bool) {
        let i = Int(self.n_tokens)
        self.token[i] = token
        self.pos[i] = position
        self.n_seq_id[i] = Int32(ids.count)
        if let seq_id = self.seq_id[i] {
            for (j, id) in ids.enumerated() {
                seq_id[j] = Int32(id)
            }
        }
        self.logits[i] = logit ? 1 : 0
        self.n_tokens += 1
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

extension Token {
    enum Kind {
        case end
        case couldBeEnd
        case normal
    }
}

public enum Role {
    case user
    case bot
}

public struct Template {
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
    
    public var preProcess: (_ input: String, _ history: [Chat]) -> String {
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
            prefix: "<s>[INST] ",
            system: ("<<SYS>>\n", "\n<</SYS>>\n\n"),
            user: ("", " [/INST]"),
            bot: (" ", "</s><s>[INST] "),
            stopSequence: "</s>",
            systemPrompt: systemPrompt,
            shouldDropLast: true
        )
    }
    
    public static let mistral = Template(
        prefix: "<s>",
        user: ("[INST] ", " [/INST]"),
        bot: ("", "</s> "),
        stopSequence: "</s>",
        systemPrompt: nil
    )
}
