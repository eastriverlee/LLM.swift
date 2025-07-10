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
            path: "llama.cpp/llama.xcframework"
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
            dependencies: ["llama", "LLMMacros"],
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
    ]
)
