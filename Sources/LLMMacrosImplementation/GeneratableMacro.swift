import SwiftCompilerPlugin
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public struct GeneratableMacro: MemberMacro, ExtensionMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingMembersOf declaration: some DeclGroupSyntax,
        conformingTo: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        let members = declaration.memberBlock.members
        let variableDecls = members.compactMap { $0.decl.as(VariableDeclSyntax.self) }
        let properties = variableDecls.flatMap { $0.bindings }.compactMap { binding -> (name: String, type: String)? in
            guard let name = binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text else {
                return nil
            }

            guard let type = binding.typeAnnotation?.type else {
                return nil
            }

            var typeName: String?
            if let simpleType = type.as(IdentifierTypeSyntax.self) {
                typeName = simpleType.name.text
            } else if let optionalType = type.as(OptionalTypeSyntax.self),
                      let wrappedType = optionalType.wrappedType.as(IdentifierTypeSyntax.self) {
                typeName = wrappedType.name.text
            }

            guard let finalTypeName = typeName else { return nil }

            switch finalTypeName {
            case "String":
                return (name, "string")
            case "Int":
                return (name, "integer")
            case "Double", "Float":
                return (name, "number")
            case "Bool":
                return (name, "boolean")
            default:
                return nil
            }
        }

        let propertiesString = properties.map { #"      "\#($0.name)": { "type": "\#($0.type)" }"# }.joined(separator: ",\n")
        let requiredString = properties.map { #"      "\#($0.name)""# }.joined(separator: ",\n")

        return [
            """
            public static var jsonSchema: String {
                return \"\"\"
                {
                  "type": "object",
                  "properties": {
                    \(raw: propertiesString)
                  },
                  "required": [
                    \(raw: requiredString)
                  ]
                }
                \"\"\"
            }
            """
        ]
    }
    
    public static func expansion(
        of node: AttributeSyntax,
        attachedTo declaration: some DeclGroupSyntax,
        providingExtensionsOf type: some TypeSyntaxProtocol,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [ExtensionDeclSyntax] {
        let typeName = type.trimmed
        
        let extensionDecl = try ExtensionDeclSyntax("extension \(typeName): Codable, Generatable") {
        }
        
        return [extensionDecl]
    }
}

enum MacroError: Error, CustomStringConvertible {
    case notAStruct
    var description: String {
        switch self {
        case .notAStruct: "Can only be applied to a struct."
        }
    }
} 