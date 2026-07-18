# ``LLM``

A Swift library providing an easy-to-use interface for running Large Language Models locally using llama.cpp.

## Overview

LLM.swift wraps llama.cpp to provide a native Swift interface for running large language models on Apple platforms, with hardware acceleration when available. Conversations are rendered with the chat template embedded in the gguf file itself, and function calling uses each model's native tool-call format.

### Key Features

- **Native Swift API** - ObservableObject integration for SwiftUI
- **Cross-platform** - Supports iOS, macOS, tvOS, and visionOS
- **Hardware acceleration** - Utilizes Metal when available
- **Embedded chat templates** - Rendered by llama.cpp's own Jinja engine, with manual `Template` override
- **Function calling** - Declare tools with the `Tool` protocol and `@Generatable` argument types
- **Structured output** - Grammar-constrained JSON generation from `@Generatable` schemas
- **Streaming responses** - Real-time token generation via AsyncStream, with thinking output separated
- **Model management** - Direct downloads from Hugging Face

## Topics

### Core Classes

- ``LLM``
- ``LLMCore``

### Model Management

- ``HuggingFaceModel``

### Chat System

- ``Chat``
- ``Role``
- ``Template``
- ``ThinkingMode``

### Function Calling

- ``Tool``
- ``ToolCall``
- ``ToolError``

### Structured Output

- ``StructuredOutput``
- ``StructuredOutputError``

### Utilities

- ``Token``
- ``Batch``
- ``Model``
- ``Vocab``
