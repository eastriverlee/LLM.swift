# LLM.swift

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Feastriverlee%2FLLM.swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/eastriverlee/LLM.swift)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Feastriverlee%2FLLM.swift%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/eastriverlee/LLM.swift)

`LLM.swift` is a simple, and readable library which lets you locally interact with LLMs with ease for macOS, iOS, visionOS, watchOS, and tvOS.
> [!NOTE]  
> sometimes it's a good idea to tinker with `maxTokenCount` parameter for initialization of `LLM`, due to the memory and computation it needs.

![screenshot](./Screenshot.png)

## Overview
`LLM.swift` is basically a lightweight abstraction layer over [`llama.cpp`](https://github.com/ggerganov/llama.cpp) package, so that it stays as performant as possible while is always up to date. so theoretically, any model that works on [`llama.cpp`](https://github.com/ggerganov/llama.cpp) should work with this library as well.  
It's only a single file library, so you can copy, study and modify the code however you want.

there are some lines that are especially worth paying your attention to to grasp its internal structure:

```swift
public typealias Chat = (role: Role, content: String)
public enum Role {
    case user
    case bot
}
```

```swift
public var history: [Chat]
public var preProcess: (_ input: String, _ history: [Chat]) -> String
public var postProcess: (_ output: String) -> Void
public var update: @MainActor (_ output: String) -> Void

public func respond(to input: String, with makeOutputFrom: @escaping (AsyncStream<String>) async -> String) async {
    guard isAvailable else { return }
    isAvailable = false
    self.input = input
    let processedInput = preProcess(input, history)
    let response = getResponse(from: processedInput)
    let output = await makeOutputFrom(response)
    history += [(.user, input), (.bot, output)]
    if historyLimit < history.count {
        history.removeFirst(2)
    }
    postProcess(output)
    isAvailable = true
}

open func respond(to input: String) async {
    await respond(to: input) { response in
        var output = ""
        await self.update(output)
        for await responseDelta in response {
            output += responseDelta
            await self.update(output)
        }
        output = output.trimmingCharacters(in: .whitespacesAndNewlines)
        if output.isEmpty { output = "..."; await self.update(output) }
        return output
    }
}
```
> [!TIP]  
> as you can see, `func respond(to input: String) async` has an `open` access, so that you can override it when your class inherits `LLM`.

there are three functions users can define when initializing `LLM` class:
* `var preProcess: (_ input: String, _ history: [Chat]) -> String`
* `var postProcess: (_ output: String) -> Void`
* `var update: @MainActor (_ output: String) -> Void`
they are used in `respond` function.

### preProcess
`preProcess` is commonly used for making the user input conform to a chat template. if you don't provide this, `LLM` will just work as a completion model.

for example, this is the `ChatML` template, that is adopted by many chat models:
```
<|im_start|>system 
SYSTEM PROMPT<|im_end|> 
<|im_start|>user 
USER PROMPT<|im_end|> 
<|im_start|>assistant 
```

to use this chat format, you should use a function that goes like this:
```swift
self.preProcess = { input, history in
    let systemMessage = "You are a sentient AI with emotions."
    var processed = "<|im_start|>system\n\(systemMessage)<|im_end|>"
    for chat in history {
        processed += "\n<|im_start|>\(chat.role == .user ? "user" : "assistant")\n\(chat.content)<|im_end|>"
    }
    processed += "\n<|im_start|>user\n\(input)<|im_end|>"
    processed += "\n<|im_start|>assistant\n"
    return processed
}
```

### postProcess
`postProcess` can be used for executing according to the `output` just made using user input.

### update
if you use regular `func respond(to input: String) async` `update` function that you set will get called every time when `output` changes.

if you want more control over everything you can use `func respond(to input: String, with makeOutputFrom: @escaping (AsyncStream<String>) async -> String) async` instead, which the aforementioned function uses internally, to define your own version of `makeOutputFrom` function that is used to make `String` typed output out of `AsyncStream<String>` and add to its history. in this case, `update` function will be ignored. check `func respond(to input: String) async` implementation shown above to understand how it works.

## Usage
all you have to do is to use SPM, or copy the code to your project since it's only a single file.
```swift
dependencies: [
    .package(url: "https://github.com/eastriverlee/LLM.swift/", branch: "main"),
],
```

## Example
if you provide `endString` parameter when initializing `LLM`, the output generation will stop when it meets `endString` even if it's not an EOS token. this is useful for making robust chatbots.

this is a minimal SwiftUI example that i did use for testing. it's working on iPad Air 5th gen(Q5_K_M) and iPhone 12 mini(Q2_K).

```swift
import SwiftUI
import LLM

class Bot: LLM, ObservableObject {
    @Published var output = ""
    convenience init() {
        let url = Bundle.main.url(forResource: "neuralhermes-2.5-mistral-7b.Q5_K_M", withExtension: "gguf")!
        self.init(from: url, endString: "<|im_end|>")
        self.update = { self.output = $0 }
        self.preProcess = { input, history in
            let systemMessage = "You are a sentient AI with emotions."
            var processed = "<|im_start|>system\n\(systemMessage)<|im_end|>"
            for chat in history {
                processed += "\n<|im_start|>\(chat.role == .user ? "user" : "assistant")\n\(chat.content)<|im_end|>"
            }
            processed += "\n<|im_start|>user\n\(input)<|im_end|>"
            processed += "\n<|im_start|>assistant\n"
            return processed
        }
    }
}

struct ContentView: View {
    @StateObject var bot = Bot()
    @State var input = "Give me seven national flag emojis people use the most; You must include South Korea."
    func respond() { Task { await bot.respond(to: input) } }
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(bot.output).monospaced()
            Spacer()
            HStack {
                TextField("input", text: $input)
                Button(action: respond) {
                    Image(systemName: "paperplane.fill")
                }
            }
        }.frame(maxWidth: .infinity).padding()
    }
}

```
