#!/bin/bash

set -e

LLAMA_CPP_DIR="llama.cpp"
LLAMA_SOURCE_DIR=".llama.cpp"

echo "Removing existing temporary source directory if it exists..."
rm -rf "$LLAMA_SOURCE_DIR"

echo "Cloning latest llama.cpp into $LLAMA_SOURCE_DIR..."
git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_SOURCE_DIR"

echo "Building XCFramework..."
cd "$LLAMA_SOURCE_DIR"
./build-xcframework.sh
cd ..

echo "Removing existing $LLAMA_CPP_DIR directory..."
rm -rf "$LLAMA_CPP_DIR"

echo "Creating $LLAMA_CPP_DIR directory..."
mkdir -p "$LLAMA_CPP_DIR"

echo "Moving XCFramework to $LLAMA_CPP_DIR..."
mv "$LLAMA_SOURCE_DIR/build-apple/llama.xcframework" "$LLAMA_CPP_DIR/"

echo "Cleaning up temporary source directory $LLAMA_SOURCE_DIR..."
rm -rf "$LLAMA_SOURCE_DIR"

echo "Running Swift tests..."
swift test