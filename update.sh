#!/bin/bash

set -e

GITHUB_REPO="ggml-org/llama.cpp"
ASSET_PATTERN="-xcframework.zip"
VENDOR_DIR="Sources/LlamaChat/vendor"

echo "Finding the latest release for $GITHUB_REPO..."
RELEASE_JSON=$(curl -s https://api.github.com/repos/$GITHUB_REPO/releases/latest)
RELEASE_TAG=$(echo "$RELEASE_JSON" | jq -r '.tag_name')
ASSET_URL=$(echo "$RELEASE_JSON" | jq -r --arg pattern "$ASSET_PATTERN" '.assets[] | select(.name | endswith($pattern)) | .browser_download_url')

if [ -z "$RELEASE_TAG" ] || [ "$RELEASE_TAG" = "null" ]; then
  echo "Error: Could not determine the latest release tag for $GITHUB_REPO."
  exit 1
fi

if [ -z "$ASSET_URL" ]; then
  echo "Error: Could not find a release asset matching '$ASSET_PATTERN' in the latest release of $GITHUB_REPO."
  exit 1
fi

echo "Latest release is $RELEASE_TAG."

WORK_DIR=$(mktemp -d)
ARCHIVE_PATH="$WORK_DIR/$(basename "$ASSET_URL")"

echo "Downloading $(basename "$ASSET_URL") to compute its checksum..."
curl -L "$ASSET_URL" -o "$ARCHIVE_PATH"

CHECKSUM=$(swift package compute-checksum "$ARCHIVE_PATH")
echo "Checksum is $CHECKSUM."

echo "Updating the llama binary target in Package.swift..."
sed -i '' -E "s|url: \"https://github.com/$GITHUB_REPO/releases/download/[^\"]+\"|url: \"$ASSET_URL\"|" Package.swift
sed -i '' -E "s|checksum: \"[a-f0-9]{64}\"|checksum: \"$CHECKSUM\"|" Package.swift

SOURCE_ARCHIVE_URL="https://github.com/$GITHUB_REPO/archive/refs/tags/$RELEASE_TAG.tar.gz"
SOURCE_ARCHIVE_PATH="$WORK_DIR/llama.cpp-$RELEASE_TAG.tar.gz"

echo "Downloading source tarball for $RELEASE_TAG from $SOURCE_ARCHIVE_URL..."
curl -L "$SOURCE_ARCHIVE_URL" -o "$SOURCE_ARCHIVE_PATH"

echo "Extracting source tarball..."
tar -xzf "$SOURCE_ARCHIVE_PATH" -C "$WORK_DIR"

SOURCE_DIR=$(command find "$WORK_DIR" -maxdepth 1 -type d -name "llama.cpp-*")

copyVendoredFile() {
  SOURCE_FILE="$1"
  DESTINATION_FILE="$2"
  if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: $SOURCE_FILE not found in the $RELEASE_TAG source tree. Upstream layout has changed; update the vendoring file list in update.sh."
    exit 1
  fi
  cp "$SOURCE_FILE" "$DESTINATION_FILE"
}

echo "Refreshing vendored chat sources in $VENDOR_DIR..."
rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR/jinja" "$VENDOR_DIR/nlohmann"

COMMON_FILE_NAMES="chat.cpp chat.h chat-auto-parser-generator.cpp chat-auto-parser-helpers.cpp chat-auto-parser-helpers.h chat-auto-parser.h chat-diff-analyzer.cpp chat-peg-parser.cpp chat-peg-parser.h peg-parser.cpp peg-parser.h json-schema-to-grammar.cpp json-schema-to-grammar.h log.cpp log.h unicode.cpp unicode.h common.h build-info.h"
for FILE_NAME in $COMMON_FILE_NAMES; do
  copyVendoredFile "$SOURCE_DIR/common/$FILE_NAME" "$VENDOR_DIR/$FILE_NAME"
done

JINJA_FILE_NAMES="caps.cpp caps.h lexer.cpp lexer.h parser.cpp parser.h runtime.cpp runtime.h string.cpp string.h utils.h value.cpp value.h"
for FILE_NAME in $JINJA_FILE_NAMES; do
  copyVendoredFile "$SOURCE_DIR/common/jinja/$FILE_NAME" "$VENDOR_DIR/jinja/$FILE_NAME"
done

NLOHMANN_FILE_NAMES="json.hpp json_fwd.hpp"
for FILE_NAME in $NLOHMANN_FILE_NAMES; do
  copyVendoredFile "$SOURCE_DIR/vendor/nlohmann/$FILE_NAME" "$VENDOR_DIR/nlohmann/$FILE_NAME"
done

INCLUDE_FILE_NAMES="llama-cpp.h"
for FILE_NAME in $INCLUDE_FILE_NAMES; do
  copyVendoredFile "$SOURCE_DIR/include/$FILE_NAME" "$VENDOR_DIR/$FILE_NAME"
done

echo "Generating framework-forwarding shim headers..."
FORWARDING_HEADER_NAMES="llama.h ggml.h ggml-alloc.h ggml-backend.h ggml-blas.h ggml-cpu.h ggml-metal.h ggml-opt.h gguf.h"
for HEADER_NAME in $FORWARDING_HEADER_NAMES; do
  echo "#include <llama/$HEADER_NAME>" > "$VENDOR_DIR/$HEADER_NAME"
done

echo "Cleaning up..."
rm -rf "$WORK_DIR"

echo "Running Swift tests..."
swift test
