set(LLVM_TARGET_DEFINITIONS TfheRustDialect.td)
mlir_tablegen(TfheRustDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TfheRustDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRTfheRustIncGen)

set(LLVM_TARGET_DEFINITIONS TfheRustOps.td)
mlir_tablegen(TfheRustOps.h.inc -gen-op-decls)
mlir_tablegen(TfheRustOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTfheRustOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TfheRustTypes.td)
mlir_tablegen(TfheRustTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TfheRustTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRTfheRustTypesIncGen)
