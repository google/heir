set(LLVM_TARGET_DEFINITIONS Comb.td) #Dialect
mlir_tablegen(CombDialect.h.inc -gen-dialect-decls -dialect=comb)
mlir_tablegen(CombDialect.cpp.inc -gen-dialect-defs -dialect=comb)
add_public_tablegen_target(MLIRCombIncGen)

set(LLVM_TARGET_DEFINITIONS Comb.td) #CombOps
mlir_tablegen(Comb.h.inc -gen-op-decls --dialect=comb)
mlir_tablegen(Comb.cpp.inc -gen-op-defs --dialect=comb)
add_public_tablegen_target(MLIRCombOpsIncGen)

set(LLVM_TARGET_DEFINITIONS Comb.td)
mlir_tablegen(CombEnums.h.inc -gen-enum-decls -dialect=comb)
mlir_tablegen(CombEnums.cpp.inc -gen-enum-defs -dialect=comb)
add_public_tablegen_target(MLIRCombDialectEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS Comb.td)
mlir_tablegen(CombTypes.h.inc -gen-typedef-decls -dialect=comb)
mlir_tablegen(CombTypes.cpp.inc -gen-typedef-defs -dialect=comb)
add_public_tablegen_target(MLIRCombTypesIncGen)
