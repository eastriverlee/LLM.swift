// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "LLM",
    platforms: [
        .macOS(.v14),
        .iOS(.v14),
        .visionOS(.v1),
        .watchOS(.v4),
        .tvOS(.v14)
    ],
    products: [
        .library(
            name: "LLM",
            targets: ["LLM"]),
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            path: "llama.cpp/llama.xcframework"
        ),
        .target(
            name: "LLM",
            dependencies: [
                "llama"
            ]),
        .testTarget(
            name: "LLMTests",
            dependencies: ["LLM"]),
    ]
)
