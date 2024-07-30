set(LLVM_TARGET_DEFINITIONS OpenfheDialect.td)
mlir_tablegen(OpenfheDialect.h.inc -gen-dialect-decls)# -dialect=Openfhe)
mlir_tablegen(OpenfheDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIROpenfheIncGen)

set(LLVM_TARGET_DEFINITIONS OpenfheOps.td)
mlir_tablegen(OpenfheOps.h.inc -gen-op-decls -dialect=openfhe)
mlir_tablegen(OpenfheOps.cpp.inc -gen-op-defs -dialect=openfhe)
add_public_tablegen_target(MLIROpenfheOpsIncGen)

set(LLVM_TARGET_DEFINITIONS OpenfheTypes.td)
mlir_tablegen(OpenfheTypes.h.inc -gen-typedef-decls -dialect=openfhe)
mlir_tablegen(OpenfheTypes.cpp.inc -gen-typedef-defs -dialect=openfhe)
add_public_tablegen_target(MLIROpenfheTypesIncGen)
