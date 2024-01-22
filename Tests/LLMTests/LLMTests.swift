import XCTest
import PowerAssert
@testable import LLM

final class LLMTests: XCTestCase {
    let systemPrompt = "You are a human."
    let userPrompt = "Are you a human or an AI?"
    let history = [Chat(.user, "Hey."), Chat(.bot, "Hi.")]
    
    func testChatMLPreProcessorWithoutSystemMessage() throws {
        let template = Template.chatML()
        let expected = """
        <|im_start|>user
        \(userPrompt)<|im_end|>
        <|im_start|>assistant
        
        """
        let output = template.preProcess(userPrompt, [])
        #assert(expected == output)
    }
    
    func testChatMLPreProcessorWithoutHistory() throws {
        let template = Template.chatML(systemPrompt)
        let expected = """
        <|im_start|>system
        \(systemPrompt)<|im_end|>
        <|im_start|>user
        \(userPrompt)<|im_end|>
        <|im_start|>assistant
        
        """
        let output = template.preProcess(userPrompt, [])
        #assert(expected == output)
    }
    
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
        let output = template.preProcess(userPrompt, history)
        #assert(expected == output)
    }
    
    func testAlpacaPreProcessorWithoutSystemMessage() throws {
        let template = Template.alpaca()
        let expected = """
        ### Instruction:
        \(userPrompt)
        
        ### Response:
        
        """
        let output = template.preProcess(userPrompt, [])
        #assert(expected == output)
    }
    
    func testAlpacaPreProcessorWithoutHistory() throws {
        let template = Template.alpaca(systemPrompt)
        let expected = """
        \(systemPrompt)

        ### Instruction:
        \(userPrompt)
        
        ### Response:
        
        """
        let output = template.preProcess(userPrompt, [])
        #assert(expected == output)
    }
    
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
        let output = template.preProcess(userPrompt, history)
        #assert(expected == output)
    }
    
    func testLLaMaPreProcessorWithoutSystemMessage() throws {
        let template = Template.llama()
        let expected = """
        <s>[INST] \(userPrompt) [/INST]
        """
        let output = template.preProcess(userPrompt, [])
        #assert(expected == output)
    }
    
    func testLLaMaPreProcessorWithoutHistory() throws {
        let template = Template.llama(systemPrompt)
        let expected = """
        <s>[INST] <<SYS>>
        \(systemPrompt)
        <</SYS>>

        \(userPrompt) [/INST]
        """
        let output = template.preProcess(userPrompt, [])
        #assert(expected == output)
    }
    
    func testLLaMaPreProcessorWithHistory() throws {
        let template = Template.llama(systemPrompt)
        let expected = """
        <s>[INST] <<SYS>>
        \(systemPrompt)
        <</SYS>>

        \(history[0].content) [/INST] \(history[1].content)</s><s>[INST] \(userPrompt) [/INST]
        """
        let output = template.preProcess(userPrompt, history)
        #assert(expected == output)
    }
}
