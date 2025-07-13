import SwiftSyntaxMacros

public protocol Generatable: Codable {
    static var jsonSchema: String { get }
}

@attached(member, names: named(jsonSchema), named(init), named(encode))
@attached(extension, conformances: Codable, Generatable, CaseIterable)
public macro Generatable() = #externalMacro(module: "LLMMacrosImplementation", type: "GeneratableMacro") 