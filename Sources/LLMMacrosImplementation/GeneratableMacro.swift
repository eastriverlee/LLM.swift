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
        if let enumDecl = declaration.as(EnumDeclSyntax.self) {
            return expandEnum(enumDecl)
        }
        
        let members = declaration.memberBlock.members
        let variableDecls = members.compactMap { $0.decl.as(VariableDeclSyntax.self) }
        let props: [(name: String, schema: String, isRequired: Bool)] = variableDecls.flatMap { $0.bindings }.compactMap { binding in
            guard let name = binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text else {
                return nil
            }
            guard let typeAnno = binding.typeAnnotation else {
                return nil
            }
            let type = typeAnno.type
            let wrappedType = wrappedTypeIfOptional(type)
            let isOptional = wrappedType != nil
            let unwrappedType = wrappedType ?? type
            let schemaExpr = schema(for: unwrappedType, in: declaration, context: context)
            return (name, schemaExpr, !isOptional)
        }
        let emptyStringToken = "\"\""
        let propertyExprsString = props.map { "\"\\\"" + $0.name + "\\\": \" + " + $0.schema }.joined(separator: " + \",\" + ")
        let propertyExprsStringFinal = propertyExprsString.isEmpty ? emptyStringToken : propertyExprsString
        let requiredExprString = props.filter { $0.isRequired }.map { "\"\\\"" + $0.name + "\\\"\"" }.joined(separator: " + \",\" + ")
        let requiredExprStringFinal = requiredExprString.isEmpty ? emptyStringToken : requiredExprString
        return [
            """
            public static var jsonSchema: String {
                return "{ \\"type\\": \\"object\\", \\"properties\\": {" + \(raw: propertyExprsStringFinal) + "}, \\"required\\": [" + \(raw: requiredExprStringFinal) + "], \\"additionalProperties\\": false }"
            }
            """
        ]
    }
    
    private static func expandEnum(_ enumDecl: EnumDeclSyntax) -> [DeclSyntax] {
        let cases = extractCases(from: enumDecl)
        let enumValues = cases.map { "\"\($0)\"" }.joined(separator: ", ")
        
        return [
            """
            public init(from decoder: Decoder) throws {
                let container = try decoder.singleValueContainer()
                let stringValue = try container.decode(String.self)
                switch stringValue {
                \(raw: cases.map { "case \"\($0)\": self = .\($0)" }.joined(separator: "\n                "))
                default: throw DecodingError.dataCorrupted(DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unknown enum value: \\(stringValue)"))
                }
            }
            
            public func encode(to encoder: Encoder) throws {
                var container = encoder.singleValueContainer()
                switch self {
                \(raw: cases.map { "case .\($0): try container.encode(\"\($0)\")" }.joined(separator: "\n                "))
                }
            }
            
            public static var jsonSchema: String {
                return \"\"\"
                {
                  "type": "string",
                  "enum": [\(raw: enumValues)]
                }
                \"\"\"
            }
            """
        ]
    }

    private static func schema(for type: TypeSyntax, in declaration: some DeclGroupSyntax, context: some MacroExpansionContext) -> String {
        if let wrappedType = wrappedTypeIfOptional(type) {
            return schema(for: wrappedType, in: declaration, context: context)
        }
        if let arrayType = type.as(ArrayTypeSyntax.self) {
            let elementSchema = schema(for: arrayType.element, in: declaration, context: context)
            return "\"{ \\\"type\\\": \\\"array\\\", \\\"items\\\": \" + " + elementSchema + " + \" }\""
        }
        if let identifierType = type.as(IdentifierTypeSyntax.self) {
            let typeName = identifierType.name.text
            if typeName == "Array", let elementType = singleGenericArgument(of: identifierType) {
                let elementSchema = schema(for: elementType, in: declaration, context: context)
                return "\"{ \\\"type\\\": \\\"array\\\", \\\"items\\\": \" + " + elementSchema + " + \" }\""
            }
            if let primitive = primitiveSchema(for: typeName) {
                return "\"{ \\\"type\\\": \\\"" + primitive + "\\\" }\""
            }
            
            return typeName + ".jsonSchema"
        }
        return "\"{}\""
    }

    private static func wrappedTypeIfOptional(_ type: TypeSyntax) -> TypeSyntax? {
        if let optionalType = type.as(OptionalTypeSyntax.self) {
            return optionalType.wrappedType
        }
        guard let identifierType = type.as(IdentifierTypeSyntax.self),
              identifierType.name.text == "Optional" else {
            return nil
        }
        return singleGenericArgument(of: identifierType)
    }

    private static func singleGenericArgument(of type: IdentifierTypeSyntax) -> TypeSyntax? {
        guard let arguments = type.genericArgumentClause?.arguments,
              arguments.count == 1 else {
            return nil
        }
        guard case .type(let argument) = arguments[arguments.startIndex].argument else {
            return nil
        }
        return argument
    }

    private static func primitiveSchema(for typeName: String) -> String? {
        switch typeName {
        case "String": return "string"
        case "Int": return "integer"
        case "Double", "Float": return "number"
        case "Bool": return "boolean"
        default: return nil
        }
    }
    
    private static func extractCases(from enumDecl: EnumDeclSyntax) -> [String] {
        var cases: [String] = []
        for member in enumDecl.memberBlock.members {
            if let caseDecl = member.decl.as(EnumCaseDeclSyntax.self) {
                for element in caseDecl.elements {
                    cases.append(element.name.text)
                }
            }
        }
        return cases
    }
    
    public static func expansion(
        of node: AttributeSyntax,
        attachedTo declaration: some DeclGroupSyntax,
        providingExtensionsOf type: some TypeSyntaxProtocol,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [ExtensionDeclSyntax] {
        let typeName = type.trimmed
        
        let conformances: String
        if declaration.is(EnumDeclSyntax.self) {
            conformances = "CaseIterable, Codable, Generatable"
        } else {
            conformances = "Codable, Generatable"
        }
        
        let extensionDecl = try ExtensionDeclSyntax("extension \(typeName): \(raw: conformances)") {
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
