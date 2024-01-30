---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug unconfirmed
assignees: eastriverlee

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
You **must** include minimal code that can reproduce the behavior, for example:
```swift
import LLM

func reproduceBug() {
    let model = HuggingFaceModel("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", .Q2_K, template: .chatML(systemPrompt))
    let bot = try await LLM(from: model, seed: 147, maxTokenCount: 16)
    let input = "this input will cause a bug"
    await bot.respond(to: input)
}
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Desktop (please complete the following information):**
- Chip: [e.g. Apple M1 Max] 
- Memory: [e.g. 32GB] 
- OS: [e.g. macOS 14.0]

**Mobile (please complete the following information):**
 - Device: [e.g. iPhone12]
 - OS: [e.g. iOS17.0]

**Additional context**
Add any other context about the problem here.
