import Foundation
import llama

public typealias Token = llama_token
public typealias Model = OpaquePointer
public typealias Chat = (role: Role, content: String)

open class LLM {
    public var model: Model
    public var history: [Chat]
    public var preProcess: (_ input: String, _ history: [Chat]) -> String
    public var postProcess: (_ output: String) -> Void
    public var update: @MainActor (_ output: String) -> Void
    
    public var topK: Int32
    public var topP: Float
    public var temp: Float
    public var historyLimit: Int
    public var path: [CChar]
    
    private var context: Context!
    private var batch: llama_batch!
    private let maxTokenCount: Int
    private let totalTokenCount: Int
    private let newlineToken: Token
    private let endString: ContiguousArray<CChar>?
    private let endStringCount: Int
    private var params: llama_context_params
    private var isFull = false
    
    public init(
        from path: String,
        endString: String? = nil,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        historyLimit: Int = 8,
        maxTokenCount: Int32 = 2048,
        preProcess: @escaping (_: String, _: [Chat]) -> String = { input, history in return input },
        postProcess: @escaping (_: String) -> Void = { print($0) },
        update: @MainActor @escaping (_: String) -> Void = { _ in }
    ) {
        self.path = path.cString(using: .utf8)!
        let model = llama_load_model_from_file(self.path, llama_model_default_params())!
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
        self.preProcess = preProcess
        self.postProcess = postProcess
        self.update = update
        self.totalTokenCount = Int(llama_n_vocab(model))
        self.newlineToken = llama_token_nl(model)
        self.endString = endString?.utf8CString
        self.endStringCount = (self.endString?.count ?? 1) - 1
        batch = llama_batch_init(Int32(self.maxTokenCount), 0, 1)
    }
    
    deinit {
        llama_free_model(model)
    }
    
    public convenience init(
        from url: URL,
        endString: String? = nil,
        history: [Chat] = [],
        seed: UInt32 = .random(in: .min ... .max),
        topK: Int32 = 40,
        topP: Float = 0.95,
        temp: Float = 0.8,
        historyLimit: Int = 8,
        maxTokenCount: Int32 = 2048,
        preProcess: @escaping (_: String, _: [Chat]) -> String = { input, history in return input },
        postProcess: @escaping (_: String) -> Void = { print($0) },
        update: @MainActor @escaping (_: String) -> Void = { _ in }
    ) {
        self.init(
            from: url.path,
            endString: endString,
            history: history,
            seed: seed,
            topK: topK,
            topP: topP,
            temp: temp,
            historyLimit: historyLimit,
            maxTokenCount: maxTokenCount,
            preProcess: preProcess,
            postProcess: postProcess,
            update: update
        )
    }
    
    private func predictNextToken() async -> Token {
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
    
    private func prepare(from input: consuming String, to output: borrowing AsyncStream<String>.Continuation) -> Bool {
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
                history.removeFirst(2)
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
        return true
    }
    
    private func finishResponse(from response: inout [String], to output: borrowing AsyncStream<String>.Continuation) async {
        multibyteCharacter.removeAll()
        var input = ""
        if 2 < history.count {
            history.removeFirst(2)
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
        let word = decode(token)
        var found = 0 < saved.endIndex
        var letters: [CChar] = []
        if let endString {
            for letter in word.utf8CString {
                guard letter != 0 else { break }
                if letter == endString[saved.endIndex] {
                    saved.endIndex += 1
                    found = true
                    saved.letters.append(letter)
                    if saved.endIndex == endStringCount {
                        saved.endIndex = 0
                        saved.letters.removeAll()
                        return false
                    }
                } else {
                    if found {
                        saved.endIndex = 0
                        if saved.letters.isEmpty {
                            output.yield(word)
                        } else {
                            output.yield(String(cString: saved.letters + [0]) + word)
                            saved.letters.removeAll()
                        }
                        return true
                    }
                    letters.append(letter)
                }
            }
        }
        if !letters.isEmpty {
            if !found {
                output.yield(word)
            } else {
                output.yield(String(cString: letters + [0]))
            }
        }
        return true
    }
    
    private func getResponse(from input: borrowing String) -> AsyncStream<String> {
        .init { output in Task {
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
    public func respond(to input: String) async {
        guard isAvailable else { return }
        isAvailable = false
        self.input = input
        let processedInput = preProcess(input, history)
        let response = getResponse(from: processedInput)
        var output = ""
        await update(output)
        for await responseDelta in response {
            output += responseDelta
            await update(output)
        }
        output = output.trimmingCharacters(in: .whitespacesAndNewlines)
        if output.isEmpty { output = "..."; await update(output) }
        history += [(.user, input), (.bot, output)]
        if historyLimit < history.count {
            history.removeFirst(2)
        }
        postProcess(output)
        isAvailable = true
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
        let addBOS = shouldAddBOS()
        let count = Int32(text.cString(using: .utf8)!.count)
        var tokenCount = count + (addBOS ? 1 : 0)
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
