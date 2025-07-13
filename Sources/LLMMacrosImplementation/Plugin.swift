import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct LLMMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        GeneratableMacro.self,
    ]
} 