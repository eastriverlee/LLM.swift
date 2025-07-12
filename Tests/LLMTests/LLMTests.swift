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
    lazy var model = HuggingFaceModel("unsloth/Qwen3-0.6B-GGUF", .Q4_K_M, template: .chatML(systemPrompt))
    let urlString = "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf?download=true"
    
    @Test
    func testRegexMatchCaseInsensitivity() async throws {
        let hasMatch = try! #"(?i)Q4_K_M"#.hasMatch(in: urlString.lowercased())
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
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-IQ4_NL.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-IQ4_XS.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q2_K.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q2_K_L.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q3_K_M.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q3_K_S.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_1.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q5_K_M.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q6_K.gguf?download=true",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf?download=true"
        ]
        #expect(urls.filter(expected.contains).count == expected.count)
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
    
    @Test
    func testEmbeddingsComparison() throws {
        let embeddings1 = Embeddings(values: [1.0, 0.0, 0.0])
        let embeddings2 = Embeddings(values: [1.0, 0.0, 0.0])
        let embeddings3 = Embeddings(values: [0.0, 1.0, 0.0])
        
        let similarity12 = embeddings1.compare(with: embeddings2)
        let similarity13 = embeddings1.compare(with: embeddings3)
        
        #expect(similarity12 == 1.0)
        #expect(similarity13 == 0.0)
    }
    
    @Test
    func testEmbeddingsMostSimilar() throws {
        let embeddings1 = Embeddings(values: [1.0, 0.0, 0.0])
        let embeddings2 = Embeddings(values: [0.9, 0.1, 0.0])
        let embeddings3 = Embeddings(values: [0.0, 1.0, 0.0])
        
        let mostSimilar = embeddings1.findMostSimilar(in: embeddings2, embeddings3)
        
        #expect(mostSimilar.values == embeddings2.values)
    }
    
    @Test
    func testGetEmbeddingsFromHuggingFaceModel() async throws {
        let bot = try await LLM(from: model)!
        
        let embeddings1 = try await bot.getEmbeddings("Hello world")
        let embeddings2 = try await bot.getEmbeddings("Hi there")
        let embeddings3 = try await bot.getEmbeddings("Goodbye")
        
        #expect(embeddings1.dimension > 0)
        #expect(embeddings1.values.count == embeddings1.dimension)
        
        let similarity = embeddings1.compare(with: embeddings2)
        #expect(similarity >= 0.0)
        #expect(similarity <= 1.0)
        
        let mostSimilar = embeddings1.findMostSimilar(in: embeddings2, embeddings3)
        #expect(mostSimilar == embeddings2 || mostSimilar == embeddings3)
    }
    
    @Test
    func testEmbeddingsDeterministicAndIndependent() async throws {
        let bot = try await LLM(from: model)!
        
        let embeddings1 = try await bot.getEmbeddings("Hello world")
        
        _ = await bot.getCompletion(from: "Tell me a joke")
        
        let embeddings2 = try await bot.getEmbeddings("Hello world")
        
        await bot.respond(to: "What is 2+2?")
        
        let embeddings3 = try await bot.getEmbeddings("Hello world")
        
        #expect(embeddings1 == embeddings2)
        #expect(embeddings2 == embeddings3)
        #expect(embeddings1 == embeddings3)
    }

    //MARK: Generatable macro tests
    
    @Generatable
    struct Person {
        let name: String
        let age: Int
        let occupation: String
        let personality: String
    }
    
    @Test
    func testStructuredOutput() async throws {
        let bot = try await LLM(from: model)!
        
        let result = try await bot.respond(
            to: "Create a person",
            as: Person.self
        )
        let person = result.value
        let output = result.rawOutput
        
        print(person)
        #expect(person.name.count > 0)
        #expect(person.age > 0)
        #expect(person.occupation.count > 0)
        #expect(person.personality.count > 0)
        #expect(!output.isEmpty)
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData)
        #expect(parsed is [String: Any])
    }

    @Generatable
    struct Book {
        let title: String
        let pages: Int
        let author: String
    }

    @Test
    func testStructuredOutputWithBook() async throws {
        let bot = try await LLM(from: model, seed: 2)!
        
        let result = try await bot.respond(
            to: "A real classic book with a title, number of pages, and an author.",
            as: Book.self
        )
        let book = result.value
        let output = result.rawOutput
        
        print(book)
        #expect(book.title.count > 0)
        #expect(book.pages > 0)
        #expect(book.author.count > 0)
        #expect(!output.isEmpty)
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData)
        #expect(parsed is [String: Any])
    }

    @Generatable
    struct Measurements {
        let height: Double
        let weight: Float
    }

    @Test
    func testStructuredOutputWithMeasurements() async throws {
        let bot = try await LLM(from: model)!
        
        let result = try await bot.respond(
            to: "Generate measurements of a random person in meters and kilograms",
            as: Measurements.self
        )
        let measurements = result.value
        let output = result.rawOutput
        
        print(measurements)
        #expect(measurements.height > 0.0)
        #expect(measurements.weight > 0.0)
        #expect(!output.isEmpty)
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData)
        #expect(parsed is [String: Any])
    }

    @Generatable
    struct Temperature {
        let degreesInCelcius: Double
    }

    @Test
    func testStructuredOutputWithSignedNumbers() async throws {
        let bot = try await LLM(from: model)!
        
        let result = try await bot.respond(
            to: "Coldest temperature below zero you can think of in Celsius, preferabably zero Kelvin in Celsius",
            as: Temperature.self
        )
        let temperature = result.value
        let output = result.rawOutput
        
        print(temperature)
        #expect(temperature.degreesInCelcius < 0)
        #expect(!output.isEmpty)
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData)
        #expect(parsed is [String: Any])
    }

    @Generatable
    struct ShoppingList {
        let items: [String]
    }

    @Test
    func testStructuredOutputWithShoppingList() async throws {
        let bot = try await LLM(from: model)!
        
        let result = try await bot.respond(
            to: "Generate a shopping list with groceries.",
            as: ShoppingList.self
        )
        let list = result.value
        let output = result.rawOutput
        
        print(list)
        #expect(!list.items.isEmpty)
        #expect(!output.isEmpty)
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData)
        #expect(parsed is [String: Any])
    }

    @Generatable
    enum Color {
        case red
        case orange
        case yellow
        case green
        case blue
        case purple
    }

    @Generatable
    struct Vegetable {
        let color: Color
        let name: String
    }

    @Test
    func testColorSchema() {
        print("Color schema: \(Color.jsonSchema)")
        print("ColoredItem schema: \(Vegetable.jsonSchema)")
        let schema = try! JSONSerialization.jsonObject(with: Vegetable.jsonSchema.data(using: .utf8)!) as! [String: Any]
        let properties = schema["properties"] as! [String: Any]
        print("properties keys: \(properties.keys)")
        let color = properties["color"] as! [String: Any]
        print("color: \(color)")
        #expect(color["enum"] as? [String] == Color.allCases.map{"\($0)"})
    }
    
    @Test
    func testStructuredOutputWithVegetable() async throws {
        let bot = try await LLM(from: model, seed: 1605617885)!
        
        let result = try await bot.respond(
            to: "Give me any vegetable that is purple.",
            as: Vegetable.self
        )
        let item = result.value
        let output = result.rawOutput
        
        print(item)
        #expect(!item.name.isEmpty)
        #expect([.red, .orange, .yellow, .green, .blue, .purple].contains(item.color))
        #expect(!output.isEmpty)
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData)
        #expect(parsed is [String: Any])
    }

    @Generatable
    struct Location {
        let latitude: Double
        let longitude: Double
    }

    @Generatable 
    struct Restaurant {
        let name: String
        let cuisine: String
        let location: Location
    }

    @Test
    func testNestedGeneratableSchemas() throws {
        // Test Location schema
        let locationSchema = try JSONSerialization.jsonObject(with: Location.jsonSchema.data(using: .utf8)!) as! [String: Any]
        #expect(locationSchema["type"] as? String == "object")
        let locationProps = locationSchema["properties"] as! [String: Any]
        #expect(locationProps.keys.contains("latitude"))
        #expect(locationProps.keys.contains("longitude"))
        let latType = locationProps["latitude"] as! [String: Any]
        #expect(latType["type"] as? String == "number")
        
        // Test Restaurant schema  
        let restaurantSchema = try JSONSerialization.jsonObject(with: Restaurant.jsonSchema.data(using: .utf8)!) as! [String: Any]
        #expect(restaurantSchema["type"] as? String == "object")
        let restaurantProps = restaurantSchema["properties"] as! [String: Any]
        #expect(restaurantProps.keys.contains("name"))
        #expect(restaurantProps.keys.contains("cuisine"))
        #expect(restaurantProps.keys.contains("location"))
        
        let locationProp = restaurantProps["location"] as! [String: Any]
        #expect(locationProp["type"] as? String == "object")
        let nestedProps = locationProp["properties"] as! [String: Any]
        #expect(nestedProps.keys.contains("latitude"))
        #expect(nestedProps.keys.contains("longitude"))
        
        // Test Garden schema
        let gardenSchema = try JSONSerialization.jsonObject(with: Garden.jsonSchema.data(using: .utf8)!) as! [String: Any]
        #expect(gardenSchema["type"] as? String == "object")
        let gardenProps = gardenSchema["properties"] as! [String: Any]
        #expect(gardenProps.keys.contains("vegetables"))
        #expect(gardenProps.keys.contains("totalPlants"))
        
        let vegetablesProp = gardenProps["vegetables"] as! [String: Any]
        #expect(vegetablesProp["type"] as? String == "array")
        let items = vegetablesProp["items"] as! [String: Any]
        #expect(items["type"] as? String == "object")
        let itemProps = items["properties"] as! [String: Any]
        #expect(itemProps.keys.contains("name"))
        #expect(itemProps.keys.contains("color"))
    }
    
    @Test
    func testNestedGeneratableStruct() async throws {
        let bot = try await LLM(from: model)!
        
        let result = try await bot.respond(
            to: "Create a restaurant with name, cuisine type, and location (latitude and longitude coordinates).",
            as: Restaurant.self
        )
        let restaurant = result.value
        let output = result.rawOutput
        
        print("Restaurant: \(restaurant)")
        print("Raw output: \(output)")
        
        #expect(!restaurant.name.isEmpty)
        #expect(!restaurant.cuisine.isEmpty)
        #expect(restaurant.location.latitude >= -90 && restaurant.location.latitude <= 90)
        #expect(restaurant.location.longitude >= -180 && restaurant.location.longitude <= 180)
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
        let location = parsed["location"] as! [String: Any]
        #expect(location["latitude"] is Double)
        #expect(location["longitude"] is Double)
    }

    @Generatable
    struct Garden {
        let vegetables: [Vegetable]
        let totalPlants: Int
    }

    @Test
    func testArrayOfGeneratableStructs() async throws {
        let bot = try await LLM(from: model)!
        
        let result = try await bot.respond(
            to: "Create a garden with 3 different vegetables (each with name and color of the vegetable)",
            as: Garden.self
        )
        let garden = result.value
        let output = result.rawOutput
        
        print("Garden: \(garden)")
        print("Raw output: \(output)")
        
        #expect(garden.vegetables.count >= 1)
        #expect(garden.totalPlants >= garden.vegetables.count)
        
        for vegetable in garden.vegetables {
            #expect(!vegetable.name.isEmpty)
            #expect([.red, .orange, .yellow, .green, .blue, .purple].contains(vegetable.color))
        }
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
        let vegetables = parsed["vegetables"] as! [[String: Any]]
        #expect(vegetables.count >= 1)
        
        for vegetable in vegetables {
            #expect(vegetable["name"] is String)
            #expect(vegetable["color"] is String)
        }
    }

    @Generatable
    struct Profile {
        let name: String
        let age: Int
        let bio: String?
        let nickname: String?
    }

    @Test
    func testOptionalFieldSchema() throws {
        let profileSchema = try JSONSerialization.jsonObject(with: Profile.jsonSchema.data(using: .utf8)!) as! [String: Any]
        #expect(profileSchema["type"] as? String == "object")
        
        let properties = profileSchema["properties"] as! [String: Any]
        #expect(properties.keys.contains("name"))
        #expect(properties.keys.contains("age"))
        #expect(properties.keys.contains("bio"))
        #expect(properties.keys.contains("nickname"))
        
        let required = profileSchema["required"] as! [String]
        #expect(required.contains("name"))
        #expect(required.contains("age"))
        #expect(!required.contains("bio"))
        #expect(!required.contains("nickname"))
        
        let bioField = properties["bio"] as! [String: Any]
        #expect(bioField["type"] as? String == "string")
    }

    @Test
    func testStructuredOutputWithOptionalFields() async throws {
        let bot = try await LLM(from: model, seed: 3978003299)!
        
        let result = try await bot.respond(
            to: "Create a minimal user profile with just name and age, no bio or nickname needed.",
            as: Profile.self
        )
        let profile = result.value
        let output = result.rawOutput
        
        print("Profile: \(profile)")
        print("Raw output: \(output)")
        
        #expect(!profile.name.isEmpty)
        #expect(profile.age > 0)
        
        let jsonData = output.data(using: String.Encoding.utf8)!
        let parsed = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
        #expect(parsed["name"] is String)
        #expect(parsed["age"] is Int)
    }
}
