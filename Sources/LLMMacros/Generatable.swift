import SwiftSyntaxMacros

public protocol Generatable: Codable {
    static var jsonSchema: String { get }
}

@attached(member, names: named(jsonSchema))
@attached(extension, conformances: Codable, Generatable)
public macro Generatable() = #externalMacro(module: "LLMMacrosImplementation", type: "GeneratableMacro") 