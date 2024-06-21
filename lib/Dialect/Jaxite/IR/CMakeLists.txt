set(LLVM_TARGET_DEFINITIONS JaxiteDialect.td)
mlir_tablegen(JaxiteDialect.h.inc -gen-dialect-decls -dialect=jaxite)
mlir_tablegen(JaxiteDialect.cpp.inc -gen-dialect-defs -dialect=jaxite)
add_public_tablegen_target(MLIRJaxiteIncGen)

set(LLVM_TARGET_DEFINITIONS JaxiteOps.td)
mlir_tablegen(JaxiteOps.h.inc -gen-op-decls)
mlir_tablegen(JaxiteOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRJaxiteOpsIncGen)

set(LLVM_TARGET_DEFINITIONS JaxiteTypes.td)
mlir_tablegen(JaxiteTypes.h.inc -gen-typedef-decls)
mlir_tablegen(JaxiteTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRJaxiteTypesIncGen)
