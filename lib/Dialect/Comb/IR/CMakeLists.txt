# This is a bit of an odd case and needs manual handling
# because it does not conform to our default naming conventions
add_custom_target(HEIRCombIncGen)

set(LLVM_TARGET_DEFINITIONS Comb.td) #Dialect
mlir_tablegen(CombDialect.h.inc -gen-dialect-decls -dialect=comb)
mlir_tablegen(CombDialect.cpp.inc -gen-dialect-defs -dialect=comb)
add_public_tablegen_target(HEIRCombDialectIncGen)
add_dependencies(HEIRCombDialectIncGen HEIRCombIncGen)

set(LLVM_TARGET_DEFINITIONS Comb.td) #Ops
mlir_tablegen(Comb.h.inc -gen-op-decls --dialect=comb)
mlir_tablegen(Comb.cpp.inc -gen-op-defs --dialect=comb)
add_public_tablegen_target(HEIRCombOpsIncGen)
add_dependencies(HEIRCombOpsIncGen HEIRCombIncGen)

set(LLVM_TARGET_DEFINITIONS Comb.td) #Enums
mlir_tablegen(CombEnums.h.inc -gen-enum-decls -dialect=comb)
mlir_tablegen(CombEnums.cpp.inc -gen-enum-defs -dialect=comb)
add_public_tablegen_target(HEIRCombDialectEnumsIncGen)
add_dependencies(HEIRCombDialectEnumsIncGen HEIRCombIncGen)

set(LLVM_TARGET_DEFINITIONS Comb.td) #Types
mlir_tablegen(CombTypes.h.inc -gen-typedef-decls -dialect=comb)
mlir_tablegen(CombTypes.cpp.inc -gen-typedef-defs -dialect=comb)
add_public_tablegen_target(HEIRCombTypesIncGen)
add_dependencies(HEIRCombTypesIncGen HEIRCombIncGen)


add_mlir_dialect_library(HEIRComb
    CombDialect.cpp
    CombOps.cpp

    DEPENDS
    HEIRCombIncGen

    LINK_LIBS PUBLIC
    HEIRInterfaces
)
