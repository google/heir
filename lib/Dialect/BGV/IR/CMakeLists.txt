set(LLVM_TARGET_DEFINITIONS BGVDialect.td)
mlir_tablegen(BGVDialect.h.inc -gen-dialect-decls -dialect=bgv)
mlir_tablegen(BGVDialect.cpp.inc -gen-dialect-defs -dialect=bgv)
add_public_tablegen_target(MLIRBGVIncGen)

set(LLVM_TARGET_DEFINITIONS BGVOps.td)
mlir_tablegen(BGVOps.h.inc -gen-op-decls)
mlir_tablegen(BGVOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRBGVOpsIncGen)
