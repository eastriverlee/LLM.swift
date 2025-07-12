import Foundation
import llama
@_exported import LLMMacros


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
    private var debugLastGeneratedTokens: [Token] = []
    
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
    
    
    public func encode(_ text: String, shouldAddBOS: Bool = true, special: Bool = true) -> [Token] {
        let count = Int32(text.cString(using: .utf8)!.count)
        var tokenCount = count + 1
        let cTokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(tokenCount))
        defer { cTokens.deallocate() }
        
        tokenCount = llama_tokenize(vocab, text, count, cTokens, tokenCount, shouldAddBOS, special)
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
            guard actualLength > 0 else { return "" }
        }
        
        let validBuffer = Array(buffer.prefix(actualLength))
        let bytes = validBuffer.map { UInt8(bitPattern: $0) }
        guard var decoded = String(bytes: bytes, encoding: .utf8) else { return "" }
        
        if decoded.contains("\0") {
            decoded = decoded.filter { $0 != "\0" }
        }
        
        tokenDecodeCache.setObject(decoded as NSString, forKey: NSNumber(value: token))
        
        return decoded
    }
    
    
    func prepareContext(for input: String) -> Bool {
        guard !input.isEmpty else { return false }
        
        currentTokenCount = 0
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
    
    func getEmbeddings(from input: String) throws -> [Float] {
        guard !input.isEmpty else { throw LLMError.inputTooLong }
        
        llama_set_embeddings(context, true)
        defer { llama_set_embeddings(context, false) }
        
        llama_memory_seq_rm(llama_get_memory(context), 1, -1, -1)
        
        let cleanTokens = prepareTokensForEmbeddings(from: input)
        try processBatchForEmbeddings(cleanTokens)
        
        return try extractEmbeddingsFromContext()
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
            seq_id[0] = 1
        }
        batch.logits[i] = isLogit ? 1 : 0
        batch.n_tokens += 1
    }
    
    private func extractEmbeddingsFromContext() throws -> [Float] {
        guard let embeddingsPtr = llama_get_embeddings(context) else { throw LLMError.embeddingsFailed }
        
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
    
    func generateWithConstraints(from input: String, jsonSchema: String) throws -> String {
        debugLastGeneratedTokens = []
        guard prepareContext(for: input) else { throw LLMError.contextCreationFailed }
        guard let parsedSchema = parseJSONSchema(jsonSchema) else { throw LLMError.contextCreationFailed }
        var output = ""
        
        try addToken("{", to: &output)
        
        for (index, field) in parsedSchema.requiredFields.enumerated() {
            try addToken("\"\(field.name)\":", to: &output)
            
            switch field.type {
            case .string(let allowedValues):
                try generateStringValue(into: &output, allowedValues: allowedValues)
            case .integer:
                try generateIntegerValue(into: &output)
            case .number:
                try generateFloatingPointValue(into: &output)
            case .boolean:
                try generateBooleanValue(into: &output)
            case .array(let itemType):
                let fieldSchema = findFieldSchemaInOriginal(field.name, originalSchema: jsonSchema)
                let itemSchema = fieldSchema?["items"] as? [String: Any]
                try generateArrayValue(into: &output, ofType: itemType, withItemSchema: itemSchema)
            case .object:
                let fieldSchema = findFieldSchemaInOriginal(field.name, originalSchema: jsonSchema)
                try generateObjectValue(into: &output, with: fieldSchema)
            }
            
            if index < parsedSchema.requiredFields.count - 1 {
                try addToken(",", to: &output)
            }
        }
        
        try addToken("}", to: &output)
        
        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private func addToken(_ token: Token, to output: inout String) throws {
        clearBatch()
        addToBatch(token: token, pos: currentTokenCount)
        guard llama_decode(context, batch) == 0 else {
            shouldContinuePredicting = false
            throw LLMError.decodingFailed
        }
        currentTokenCount += 1
        output += decode(token)
        debugLastGeneratedTokens.append(token)
    }
    
    private func addToken(_ string: String, to output: inout String) throws {
        let tokens = encode(string, shouldAddBOS: false, special: false)
        for token in tokens {
            try addToken(token, to: &output)
        }
    }
    
    private func generateStringValue(into output: inout String, allowedValues: [String]? = nil) throws {
        if let allowedValues, !allowedValues.isEmpty {
            try generateConstrainedString(into: &output, allowedValues: allowedValues)
            return
        }

        try addToken("\"", to: &output)
        
        let maxTokens = 32
        var tokensInString: [Token] = []
        let quoteToken = encode("\"", shouldAddBOS: false, special: false).first!
        
        var hasContent = false
        
        while tokensInString.count < maxTokens {
            var allowedTokens = getStringTokensForField()
            if hasContent {
                allowedTokens.insert(quoteToken)
            } else {
                allowedTokens.remove(encode(" ", shouldAddBOS: false, special: false).first!)
            }
            
            let predictedToken = sampleNextToken(from: allowedTokens)
            guard predictedToken != endToken else { break }
            
            if predictedToken == quoteToken {
                try addToken(predictedToken, to: &output)
                return
            }
            
            try addToken(predictedToken, to: &output)
            let decoded = decode(predictedToken)
            tokensInString.append(predictedToken)
            
            if !decoded.trimmingCharacters(in: .whitespaces).isEmpty {
                hasContent = true
            }
            
            if tokensInString.count > 2 && tokensInString.last == tokensInString[tokensInString.count - 2] {
                break
            }
        }
        
        if !output.hasSuffix("\"") {
            try addToken("\"", to: &output)
        }
    }
    
    private func generateConstrainedString(into output: inout String, allowedValues: [String]) throws {
        try addToken("\"", to: &output)
        
        var currentTokens: [Token] = []
        var candidateValues = allowedValues
        
        while !candidateValues.isEmpty {
            let possibleNextTokens = Set(candidateValues.flatMap { value in
                let fullTokens = encode(value, shouldAddBOS: false, special: false)
                return currentTokens.count < fullTokens.count ? [fullTokens[currentTokens.count]] : []
            })
            
            let quoteToken = encode("\"", shouldAddBOS: false, special: false).first!
            let allowedTokens = possibleNextTokens.union([quoteToken])
            
            let nextToken = sampleNextToken(from: allowedTokens)
            
            if nextToken == quoteToken {
                try addToken(nextToken, to: &output)
                return
            }
            
            currentTokens.append(nextToken)
            try addToken(nextToken, to: &output)
            
            candidateValues = candidateValues.filter { value in
                let valueTokens = encode(value, shouldAddBOS: false, special: false)
                return currentTokens.count <= valueTokens.count && 
                       Array(valueTokens.prefix(currentTokens.count)) == currentTokens
            }
        }
        
        try addToken("\"", to: &output)
    }
    
    private func generateIntegerValue(into output: inout String) throws {
        let maxLength = 19
        var generatedString = ""
        let digitTokens = Set("0123456789".compactMap { encode(String($0), shouldAddBOS: false, special: false).first })
        let minusToken = encode("-", shouldAddBOS: false, special: false).first!
        let commaToken = encode(",", shouldAddBOS: false, special: false).first!
        let closingBraceToken = encode("}", shouldAddBOS: false, special: false).first!
        let terminators = Set([commaToken, closingBraceToken])

        guard !digitTokens.isEmpty else { throw LLMError.tokenizationFailed }

        while generatedString.count < maxLength {
            var allowedTokens: Set<Token>

            if generatedString.isEmpty {
                allowedTokens = digitTokens.union([minusToken])
            } else if generatedString == "-" {
                allowedTokens = digitTokens
            } else if generatedString == "0" || generatedString == "-0" {
                allowedTokens = terminators
            } else {
                allowedTokens = digitTokens.union(terminators)
            }

            let nextToken = sampleNextToken(from: allowedTokens)
            guard nextToken != endToken else { break }

            if terminators.contains(nextToken) {
                break
            }

            try addToken(nextToken, to: &output)
            generatedString += decode(nextToken)
        }

        if generatedString.isEmpty || generatedString == "-" {
            let zeroToken = encode("0", shouldAddBOS: false, special: false).first!
            try addToken(zeroToken, to: &output)
        }
    }
    
    private func generateFloatingPointValue(into output: inout String) throws {
        let maxLength = 10
        var generatedString = ""
        let digitTokens = Set("0123456789".compactMap { encode(String($0), shouldAddBOS: false, special: false).first })
        let minusToken = encode("-", shouldAddBOS: false, special: false).first!
        let dotToken = encode(".", shouldAddBOS: false, special: false).first!
        let commaToken = encode(",", shouldAddBOS: false, special: false).first!
        let closingBraceToken = encode("}", shouldAddBOS: false, special: false).first!
        let terminators = Set([commaToken, closingBraceToken])

        guard !digitTokens.isEmpty else { throw LLMError.tokenizationFailed }

        while generatedString.count < maxLength {
            var allowedTokens: Set<Token>

            if generatedString.isEmpty {
                allowedTokens = digitTokens.union([minusToken])
            } else if generatedString == "-" {
                allowedTokens = digitTokens
            } else if generatedString == "0" || generatedString == "-0" {
                allowedTokens = terminators.union([dotToken])
            } else if generatedString.contains(".") {
                allowedTokens = digitTokens.union(terminators)
            } else { // integer part, e.g. "12", "-34"
                allowedTokens = digitTokens.union(terminators).union([dotToken])
            }

            let nextToken = sampleNextToken(from: allowedTokens)
            guard nextToken != endToken else { break }

            if terminators.contains(nextToken) {
                break
            }
            
            let decoded = decode(nextToken)
            if decoded == "." && generatedString.contains(".") { break }

            try addToken(nextToken, to: &output)
            generatedString += decoded
        }
        
        if generatedString.isEmpty || generatedString == "-" {
            let zeroToken = encode("0", shouldAddBOS: false, special: false).first!
            try addToken(zeroToken, to: &output)
        }

        if generatedString.last == "." {
            let zeroToken = encode("0", shouldAddBOS: false, special: false).first!
            try addToken(zeroToken, to: &output)
        }
    }
    
    private func generateBooleanValue(into output: inout String) throws {
        let trueToken = encode("true", shouldAddBOS: false, special: false).first!
        let falseToken = encode("false", shouldAddBOS: false, special: false).first!
        let allowedTokens = Set([trueToken, falseToken])
        
        let predictedToken = sampleNextToken(from: allowedTokens)
        guard predictedToken != endToken else {
            try addToken("false", to: &output)
            return
        }
        
        try addToken(predictedToken, to: &output)
    }
    
    private func generateObjectValue(into output: inout String, with schema: [String: Any]? = nil) throws {
        try addToken("{", to: &output)
        
        guard let schema = schema,
              let properties = schema["properties"] as? [String: Any],
              let required = schema["required"] as? [String] else {
            try addToken("}", to: &output)
            return
        }
        
        for (index, fieldName) in required.enumerated() {
            guard let fieldInfo = properties[fieldName] as? [String: Any] else { continue }
            
            try addToken("\"\(fieldName)\":", to: &output)
            try generateValueForFieldInfo(fieldInfo, into: &output)
            
            if index < required.count - 1 {
                try addToken(",", to: &output)
            }
        }
        
        try addToken("}", to: &output)
    }
    
    private func generateValueForFieldInfo(_ fieldInfo: [String: Any], into output: inout String) throws {
        if let enumValues = fieldInfo["enum"] as? [String] {
            try generateStringValue(into: &output, allowedValues: enumValues)
            return
        }
        
        let typeString = fieldInfo["type"] as? String ?? ""
        switch typeString {
        case "string":
            try generateStringValue(into: &output, allowedValues: nil)
        case "integer":
            try generateIntegerValue(into: &output)
        case "number":
            try generateFloatingPointValue(into: &output)
        case "boolean":
            try generateBooleanValue(into: &output)
        case "object":
            try generateObjectValue(into: &output, with: fieldInfo)
        case "array":
            if let items = fieldInfo["items"] as? [String: Any] {
                let itemType = parseItemType(from: items)
                try generateArrayValue(into: &output, ofType: itemType, withItemSchema: items)
            } else {
                try addToken("[]", to: &output)
            }
        default:
            try addToken("null", to: &output)
        }
    }
    
    private func parseItemType(from items: [String: Any]) -> JSONFieldType {
        if let enumValues = items["enum"] as? [String] {
            return .string(allowedValues: enumValues)
        }
        
        let itemTypeString = items["type"] as? String ?? ""
        switch itemTypeString {
        case "string": return .string(allowedValues: nil)
        case "integer": return .integer
        case "number": return .number
        case "boolean": return .boolean
        case "object": return .object
        default: return .string(allowedValues: nil)
        }
    }
    
    private func findFieldSchemaInOriginal(_ fieldName: String, originalSchema: String) -> [String: Any]? {
        guard let schemaData = originalSchema.data(using: .utf8),
              let schema = try? JSONSerialization.jsonObject(with: schemaData) as? [String: Any],
              let properties = schema["properties"] as? [String: Any],
              let fieldSchema = properties[fieldName] as? [String: Any] else {
            return nil
        }
        return fieldSchema
    }
    
    private func generateArrayValue(into output: inout String, ofType itemType: JSONFieldType, withItemSchema itemSchema: [String: Any]? = nil) throws {
        try addToken("[", to: &output)
        
        let maxItems = 5
        var itemCount = 0
        
        let closingBracketToken = encode("]", shouldAddBOS: false, special: false).first!
        let commaToken = encode(",", shouldAddBOS: false, special: false).first!
        
        while itemCount < maxItems {
            switch itemType {
            case .string(let allowedValues):
                try generateStringValue(into: &output, allowedValues: allowedValues)
            case .integer:
                try generateIntegerValue(into: &output)
            case .number:
                try generateFloatingPointValue(into: &output)
            case .boolean:
                try generateBooleanValue(into: &output)
            case .object:
                try generateObjectValue(into: &output, with: itemSchema)
            default:
                break
            }
            
            itemCount += 1
            
            if itemCount >= maxItems {
                break
            }
            
            let subsequentTokens = Set([commaToken, closingBracketToken])
            let nextToken = sampleNextToken(from: subsequentTokens)
            
            try addToken(nextToken, to: &output)
            
            if nextToken == closingBracketToken {
                return
            }
        }
        
        if !output.hasSuffix("]") {
            try addToken("]", to: &output)
        }
    }
    
    // MARK: - Schema-Driven JSON Generation
    
    private enum JSONFieldType: Equatable {
        case string(allowedValues: [String]?)
        case integer
        case number
        case boolean
        case object
        indirect case array(itemType: JSONFieldType)
    }
    
    private struct SchemaField: Equatable {
        let name: String
        let type: JSONFieldType
        let required: Bool
    }
    
    private struct ParsedSchema {
        let fields: [SchemaField]
        let requiredFields: [SchemaField]
    }
    
    private func getStringTokensForField() -> Set<Token> {
        var tokens: Set<Token> = []
        
        for i in 0..<totalTokenCount {
            let token = Token(i)
            let decoded = decode(token)
            
            if !decoded.isEmpty && decoded.count <= 10 && decoded.allSatisfy({ char in
                let ascii = char.asciiValue ?? 0
                return ascii >= 32 && ascii <= 126 &&
                (char.isLetter || char == " " ||
                 ".-_'".contains(char))
            }) && !decoded.contains(where: { $0.isNumber }) {
                tokens.insert(token)
            }
        }
        
        return tokens
    }
    
    private func parseJSONSchema(_ schemaString: String) -> ParsedSchema? {
        guard let schemaData = schemaString.data(using: .utf8),
              let schema = try? JSONSerialization.jsonObject(with: schemaData) as? [String: Any],
              let properties = schema["properties"] as? [String: Any] else {
            return nil
        }
        
        let required = schema["required"] as? [String] ?? []
        var fields: [SchemaField] = []
        
        for (fieldName, fieldData) in properties {
            guard let fieldInfo = fieldData as? [String: Any] else { continue }
            
            let fieldType: JSONFieldType
            if let enumValues = fieldInfo["enum"] as? [String] {
                fieldType = .string(allowedValues: enumValues)
            } else {
                let typeString = fieldInfo["type"] as? String ?? ""
                switch typeString {
                case "string": fieldType = .string(allowedValues: nil)
                case "integer": fieldType = .integer
                case "number": fieldType = .number
                case "boolean": fieldType = .boolean
                case "object": fieldType = .object
                case "array":
                    guard let items = fieldInfo["items"] as? [String: Any],
                          let itemTypeString = items["type"] as? String else { continue }
                    switch itemTypeString {
                    case "string": 
                        if let enumValues = items["enum"] as? [String] {
                            fieldType = .array(itemType: .string(allowedValues: enumValues))
                        } else {
                            fieldType = .array(itemType: .string(allowedValues: nil))
                        }
                    case "integer": fieldType = .array(itemType: .integer)
                    case "number": fieldType = .array(itemType: .number)
                    case "object": fieldType = .array(itemType: .object)
                    case "boolean": fieldType = .array(itemType: .boolean)
                    default: continue
                    }
                default: continue
                }
            }
            
            let isRequired = required.contains(fieldName)
            fields.append(SchemaField(name: fieldName, type: fieldType, required: isRequired))
        }
        
        let requiredFields = fields.filter { $0.required }
        return ParsedSchema(fields: fields, requiredFields: requiredFields)
    }
    
    private func sampleNextToken(from allowedTokens: Set<Token>) -> Token {
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
        
        if allowedTokens.isEmpty { return token }
        
        return allowedTokens.contains(token) ? token : findBestAllowedToken(allowedTokens: allowedTokens)
    }
    
    private func findBestAllowedToken(allowedTokens: Set<Token>) -> Token {
        guard !allowedTokens.isEmpty else { return Token(32) }
        
        // If there's only one token, return it directly
        if allowedTokens.count == 1 {
            return allowedTokens.first!
        }
        
        let logits = llama_get_logits(context)!
        
        return allowedTokens.max { logits[Int($0)] < logits[Int($1)] } ?? allowedTokens.first!
    }
    
    private func isValidJSON(_ text: String) -> Bool {
        guard let data = text.data(using: .utf8) else { return false }
        do {
            _ = try JSONSerialization.jsonObject(with: data)
            return true
        } catch {
            return false
        }
    }
    
    private func isComplete(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.hasPrefix("{") && trimmed.hasSuffix("}")
    }
}

public enum LLMError: Error {
    case modelLoadFailed
    case contextCreationFailed
    case tokenizationFailed
    case decodingFailed
    case inputTooLong
    case decodeFailed
    case embeddingsFailed
}

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

public struct StructuredOutput<T: Generatable> {
    public let value: T
    public let rawOutput: String
    
    public init(value: T, rawOutput: String) {
        self.value = value
        self.rawOutput = rawOutput
    }
}

public enum StructuredOutputError: Error {
    case invalidJSON
    case schemaMismatch
    case decodingFailed
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
    
    static var isLogSilenced = false
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
    
    public func getEmbeddings(_ text: String) async throws -> Embeddings {
        let values = try await core.getEmbeddings(from: text)
        return Embeddings(values: values)
    }
    
    public func respond<T: Generatable>(
        to prompt: String,
        as type: T.Type
    ) async throws -> StructuredOutput<T> {
        let schemaPrompt = """
        \(prompt)
        
        Generate a JSON response that:
        Matches this exact schema: \(T.jsonSchema)
        Return only the JSON object, no other text.
        """
        
        let rawOutput = try await core.generateWithConstraints(
            from: schemaPrompt,
            jsonSchema: T.jsonSchema
        )
        
        guard let jsonData = rawOutput.data(using: .utf8) else {
            throw StructuredOutputError.invalidJSON
        }
        
        do {
            let decodedValue = try JSONDecoder().decode(T.self, from: jsonData)
            return StructuredOutput(value: decodedValue, rawOutput: rawOutput)
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

