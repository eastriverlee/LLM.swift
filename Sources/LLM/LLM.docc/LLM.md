# ``LLM``

A Swift library providing an easy-to-use interface for running Large Language Models locally using llama.cpp.

## Overview

LLM.swift is a single-file Swift library that wraps llama.cpp to provide a native Swift interface for running large language models on Apple platforms. It supports iOS, macOS, tvOS, watchOS, and visionOS with hardware acceleration when available.

### Key Features

- **Native Swift API** - ObservableObject integration for SwiftUI
- **Cross-platform** - Supports all Apple platforms
- **Hardware acceleration** - Utilizes Metal Performance Shaders when available
- **Template system** - Built-in support for popular chat templates
- **Streaming responses** - Real-time token generation via AsyncStream
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

### Utilities

- ``Token``
- ``Batch``
- ``Model``
- ``Vocab``