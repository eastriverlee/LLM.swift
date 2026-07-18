// swift-tools-version: 5.9
import PackageDescription
import CompilerPluginSupport

let package = Package(
    name: "LLM",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
        .watchOS(.v9),
        .tvOS(.v16),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "LLM",
            targets: ["LLM"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-syntax.git", from: "602.0.0-latest"),
        .package(url: "https://github.com/apple/swift-testing.git", branch: "main")
    ],
    targets: [
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b10068/llama-b10068-xcframework.zip",
            checksum: "5238397dd4ca305c9db537c3ae106948909ba2605e77d2d3463ac2d2ca08cc8a"
        ),
        .target(
            name: "LlamaChat",
            dependencies: ["llama"],
            path: "Sources/LlamaChat",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("vendor")
            ],
            linkerSettings: [
                .linkedLibrary("c++")
            ]
        ),
        .macro(
            name: "LLMMacrosImplementation",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax")
            ],
            path: "Sources/LLMMacrosImplementation"
        ),
        .target(
            name: "LLMMacros",
            dependencies: [
                "LLMMacrosImplementation",
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax")
            ],
            path: "Sources/LLMMacros"
        ),
        .target(
            name: "LLM",
            dependencies: ["llama", "LlamaChat", "LLMMacros"],
            path: "Sources/LLM"
        ),
        .testTarget(
            name: "LLMTests",
            dependencies: [
                "LLM",
                "LLMMacros",
                .product(name: "Testing", package: "swift-testing")
            ],
            path: "Tests/LLMTests"
        )
    ],
    cxxLanguageStandard: .cxx17
)
