import Testing
import Foundation
@testable import LLM

final class LLMTests {
    //MARK: Template tests
    let systemPrompt = "You are a human."
    let userPrompt = "Are you a human or an AI?"
    let history = [Chat(.user, "Hey."), Chat(.bot, "Hi.")]
    
    @Test
    func testChatMLPreProcessorWithoutSystemMessage() throws {
        let template = Template.chatML()
        let expected = """
        <|im_start|>user
        \(userPrompt)<|im_end|>
        <|im_start|>assistant
        
        """
        let output = template.preprocess(userPrompt, [])
        #expect(expected == output)
    }
    
    @Test
    func testChatMLPreProcessorWithoutHistory() throws {
        let template = Template.chatML(systemPrompt)
        let expected = """
        <|im_start|>system
        \(systemPrompt)<|im_end|>
        <|im_start|>user
        \(userPrompt)<|im_end|>
        <|im_start|>assistant
        
        """
        let output = template.preprocess(userPrompt, [])
        #expect(expected == output)
    }
    
    @Test
    func testChatMLPreProcessorWithHistory() throws {
        let template = Template.chatML(systemPrompt)
        let expected = """
        <|im_start|>system
        \(systemPrompt)<|im_end|>
        <|im_start|>user
        \(history[0].content)<|im_end|>
        <|im_start|>assistant
        \(history[1].content)<|im_end|>
        <|im_start|>user
        \(userPrompt)<|im_end|>
        <|im_start|>assistant
        
        """
        let output = template.preprocess(userPrompt, history)
        #expect(expected == output)
    }
    
    @Test
    func testAlpacaPreProcessorWithoutSystemMessage() throws {
        let template = Template.alpaca()
        let expected = """
        ### Instruction:
        \(userPrompt)
        
        ### Response:
        
        """
        let output = template.preprocess(userPrompt, [])
        #expect(expected == output)
    }
    
    @Test
    func testAlpacaPreProcessorWithoutHistory() throws {
        let template = Template.alpaca(systemPrompt)
        let expected = """
        \(systemPrompt)

        ### Instruction:
        \(userPrompt)
        
        ### Response:
        
        """
        let output = template.preprocess(userPrompt, [])
        #expect(expected == output)
    }
    
    @Test
    func testAlpacaPreProcessorWithHistory() throws {
        let template = Template.alpaca(systemPrompt)
        let expected = """
        \(systemPrompt)

        ### Instruction:
        \(history[0].content)
        
        ### Response:
        \(history[1].content)
        
        ### Instruction:
        \(userPrompt)
        
        ### Response:
        
        """
        let output = template.preprocess(userPrompt, history)
        #expect(expected == output)
    }
    
    @Test
    func testLLaMaPreProcessorWithoutSystemMessage() throws {
        let template = Template.llama()
        let expected = """
        [INST] \(userPrompt) [/INST]
        """
        let output = template.preprocess(userPrompt, [])
        #expect(expected == output)
    }
    
    @Test
    func testLLaMaPreProcessorWithoutHistory() throws {
        let template = Template.llama(systemPrompt)
        let expected = """
        [INST] <<SYS>>
        \(systemPrompt)
        <</SYS>>

        \(userPrompt) [/INST]
        """
        let output = template.preprocess(userPrompt, [])
        #expect(expected == output)
    }
    
    @Test
    func testLLaMaPreProcessorWithHistory() throws {
        let template = Template.llama(systemPrompt)
        let expected = """
        [INST] <<SYS>>
        \(systemPrompt)
        <</SYS>>

        \(history[0].content) [/INST] \(history[1].content)</s><s>[INST] \(userPrompt) [/INST]
        """
        let output = template.preprocess(userPrompt, history)
        #expect(expected == output)
    }
    
    @Test
    func testMistralPreProcessorWithoutHistory() throws {
        let template = Template.mistral
        let expected = """
        [INST] \(userPrompt) [/INST]
        """
        let output = template.preprocess(userPrompt, [])
        #expect(expected == output)
    }
    
    @Test
    func testMistralPreProcessorWithHistory() throws {
        let template = Template.mistral
        let expected = """
        [INST] \(history[0].content) [/INST]\(history[1].content)</s> [INST] \(userPrompt) [/INST]
        """
        let output = template.preprocess(userPrompt, history)
        #expect(expected == output)
    }
    
    //MARK: HuggingFaceModel tests
    lazy var model = HuggingFaceModel("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", .Q2_K, template: .chatML(systemPrompt))
    let urlString = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf?download=true"
    
    @Test
    func testRegexMatchCaseInsensitivity() async throws {
        let hasMatch = try! #"(?i)Q2_K"#.hasMatch(in: urlString.lowercased())
        let expected = true
        #expect(hasMatch == expected)
    }
    
    @Test
    func testFilterHasMatch() async throws {
        let hasMatch = try! model.filterRegexPattern.hasMatch(in: urlString)
        let expected = true
        #expect(hasMatch == expected)
    }
    
    @Test
    func testGetDownloadURLStringsFromHuggingFaceModel() async throws {
        let urls = try await model.getDownloadURLStrings()
        let expected = [
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_L.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_0.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf?download=true",
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true"
        ]
        #expect(urls == expected)
    }
    
    @Test
    func testGetDownloadURLFromHuggingFaceModel() async throws {
        let url = try await model.getDownloadURL()!
        let expected = URL(string: urlString)!
        #expect(url == expected)
    }
    
    @Test
    func testInitFromHuggingFaceModel() async throws {
        let bot = try await LLM(from: model)!
        #expect(!bot.path.isEmpty)
    }
    
    @Test
    func testInitializerWithTempate() async throws {
        let template = model.template
        let bot = try await LLM(from: model)!
        #expect(bot.preprocess(userPrompt, []) == template.preprocess(userPrompt, []))
    }
    
    @Test
    func testInferenceFromHuggingFaceModel() async throws {
        let bot = try await LLM(from: model)!
        bot.update = { output in
            print("...\(output ?? "nil")")
        }
        let input = "have you heard of this so-called LLM.swift library?"
        await bot.respond(to: input)
        #expect(!bot.output.isEmpty)
    }
    
    @Test
    func testEncodingDecodingFromHuggingFaceModel() async throws {
        let bot = try await LLM(from: model)!
        let input = "have you heard of this so-called LLM.swift library?"
        var tokens = await bot.core.encode(input)
        tokens.removeLast()
        var decoded = ""
        for token in tokens {
            decoded += await bot.core.decode(token)
        }
        decoded = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(!tokens.isEmpty)
        #expect(decoded == input)
    }
}
