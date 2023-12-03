import Foundation
import llama

public typealias Token = llama_token
public typealias Model = OpaquePointer
public typealias Context = OpaquePointer
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
    
    private let context: Context
    private var batch: llama_batch
    private let maxTokenCount: Int
    private let totalTokenCount: Int
    private let newlineToken: Token
    private let endTokens: [Token]
    private let endTokenCount: Int
    private var params: llama_context_params
    
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
        llama_backend_init(false)
        let model = llama_load_model_from_file(path.cString(using: .utf8), llama_model_default_params())!
        params = llama_context_default_params()
        let processorCount = UInt32(ProcessInfo().processorCount)
        self.maxTokenCount = Int(min(maxTokenCount, llama_n_ctx_train(model)))
        batch = llama_batch_init(Int32(self.maxTokenCount), 0, 1)
        params.seed = seed
        params.n_ctx = UInt32(maxTokenCount)
        params.n_batch = params.n_ctx
        params.n_threads = processorCount
        params.n_threads_batch = processorCount
        context = llama_new_context_with_model(model, params)!
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
        if let endString {
            endTokens = model.encodeOnly(endString)
            endTokenCount = endTokens.count
        } else {
            endTokens = []
            endTokenCount = 0
        }
    }
    
    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
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
        let logits = llama_get_logits_ith(context, batch.n_tokens - 1)!
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
            llama_sample_top_k(context, &candidates, topK, 1)
            llama_sample_top_p(context, &candidates, topP, 1)
            llama_sample_temp(context, &candidates, temp)
            token = llama_sample_token(context, &candidates)
        }
        batch.clear()
        batch.add(token, currentCount, [0], true)
        llama_decode(context, batch)
        return token
    }
    
    private var currentCount: Int32!
    private var decoded = ""
    
    private func checkKind(of token: Token) -> Token.Kind {
        struct endToken { static var index = 0 }
        if token == llama_token_eos(model) {
            endToken.index = 0
            return .end
        }
        if 0 < endTokenCount && endTokens[endToken.index] == token {
            endToken.index += 1
            if endTokenCount == endToken.index {
                endToken.index = 0
                return .end
            } else {
                return .couldBeEnd
            }
        }
        endToken.index = 0
        return .normal
    }
    
    private func getResponse(from input: String) -> AsyncStream<String> {
        .init { output in Task {
            var tokens = encode(input)
            let initialCount = tokens.count
            currentCount = Int32(initialCount)
            for (i, token) in tokens.enumerated() {
                batch.n_tokens = Int32(i)
                batch.add(token, batch.n_tokens, [0], i == initialCount - 1)
            }
            llama_decode(context, batch)
            tokens.removeAll(keepingCapacity: true)
            while currentCount <= maxTokenCount {
                let token = await predictNextToken()
                switch checkKind(of: token) {
                case .couldBeEnd:
                    tokens.append(token)
                case .end:
                    output.finish()
                    return
                case .normal:
                    tokens.append(token)
                    let word = decode(tokens)
                    output.yield(word)
                    tokens.removeAll(keepingCapacity: true)
                }
                currentCount += 1
            }
            output.finish()
        } }
    }
    
    public func respond(to input: String) async {
        let processedInput = preProcess(input, history)
        let response = getResponse(from: processedInput)
        var output = ""
        await update(output)
        for await responseDelta in response {
            output += responseDelta
            await update(output)
        }
        history += [(.user, input), (.bot, output)]
        if historyLimit < history.count {
            history = .init(history.dropFirst(2))
        }
        postProcess(output)
    }
    
    public func decode(_ tokens: [Token]) -> String {
        struct multibyte { static var character: [CChar] = [] }
        return tokens.reduce("", { $0 + model.decode($1, with: &multibyte.character) } )
    }
    
    @inlinable
    public func encode(_ text: String) -> [Token] {
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
    
    public func isEmpty(_ token: Token) -> Bool {
        var nothing: [CChar] = []
        return decode(token, with: &nothing).isEmpty && nothing.isEmpty
    }
    
    public func decode(_ token: Token, with multibyteCharacter: inout [CChar]) -> String {
        var bufferLength = 16
        var buffer: [CChar] = .init(repeating: 0, count: bufferLength)
        var length = llama_token_to_piece(self, token, &buffer, Int32(bufferLength))
        guard 0 != length else { return "" }
        if length < 0 {
            bufferLength *= 2
            buffer = .init(repeating: 0, count: bufferLength)
            length = llama_token_to_piece(self, token, &buffer, Int32(bufferLength))
        }
        buffer.removeLast(bufferLength - Int(length))
        if multibyteCharacter.isEmpty, let decoded = String(cString: buffer + [0], encoding: .utf8) {
            return decoded
        }
        multibyteCharacter.append(contentsOf: buffer)
        let data = Data(multibyteCharacter.map { UInt8(bitPattern: $0) })
        guard let decoded = String(data: data, encoding: .utf8) else { return "" }
        multibyteCharacter.removeAll(keepingCapacity: true)
        return decoded
    }

    public func encodeOnly(_ text: String) -> [Token] {
        let tokens = encode("." + text)
        return .init(tokens.filter({ !isEmpty($0) }).dropFirst())
    }
    
    public func encode(_ text: String) -> [Token] {
        let addBOS = shouldAddBOS()
        let count = Int32(text.cString(using: .utf8)!.count)
        var tokenCount = count + (addBOS ? 1 : 0)
        let cTokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(tokenCount)); defer { cTokens.deallocate() }
        tokenCount = llama_tokenize(self, text, count, cTokens, tokenCount, addBOS, false)
        let tokens = (0..<Int(tokenCount)).map { cTokens[$0] }
        return tokens
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
