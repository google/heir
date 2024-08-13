set(LLVM_TARGET_DEFINITIONS ModArithDialect.td)
mlir_tablegen(ModArithDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ModArithDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRModArithIncGen)

set(LLVM_TARGET_DEFINITIONS ModArithOps.td)
mlir_tablegen(ModArithOps.h.inc -gen-op-decls)
mlir_tablegen(ModArithOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRModArithOpsIncGen)
