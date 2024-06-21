set(LLVM_TARGET_DEFINITIONS ArithExtDialect.td)
mlir_tablegen(ArithExtDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ArithExtDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRArithExtIncGen)

set(LLVM_TARGET_DEFINITIONS ArithExtOps.td)
mlir_tablegen(ArithExtOps.h.inc -gen-op-decls)
mlir_tablegen(ArithExtOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRArithExtOpsIncGen)
