set(LLVM_TARGET_DEFINITIONS RandomDialect.td)
mlir_tablegen(RandomDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RandomDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRRandomIncGen)

set(LLVM_TARGET_DEFINITIONS RandomOps.td)
mlir_tablegen(RandomOps.h.inc -gen-op-decls)
mlir_tablegen(RandomOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRRandomOpsIncGen)

set(LLVM_TARGET_DEFINITIONS RandomTypes.td)
mlir_tablegen(RandomTypes.h.inc -gen-typedef-decls)
mlir_tablegen(RandomTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRRandomTypesIncGen)

set(LLVM_TARGET_DEFINITIONS RandomEnums.td)
mlir_tablegen(RandomEnums.h.inc -gen-enum-decls)
mlir_tablegen(RandomEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRRandomEnumsIncGen)
