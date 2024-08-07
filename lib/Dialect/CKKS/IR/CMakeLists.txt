set(LLVM_TARGET_DEFINITIONS CKKSDialect.td)
mlir_tablegen(CKKSDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CKKSDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRCKKSIncGen)

set(LLVM_TARGET_DEFINITIONS CKKSOps.td)
mlir_tablegen(CKKSOps.h.inc -gen-op-decls)
mlir_tablegen(CKKSOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRCKKSOpsIncGen)
