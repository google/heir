set(LLVM_TARGET_DEFINITIONS LWEDialect.td)
mlir_tablegen(LWEDialect.h.inc -gen-dialect-decls -dialect=lwe)
mlir_tablegen(LWEDialect.cpp.inc -gen-dialect-defs -dialect=lwe)
add_public_tablegen_target(MLIRLWEIncGen)

set(LLVM_TARGET_DEFINITIONS LWEOps.td)
mlir_tablegen(LWEOps.h.inc -gen-op-decls -dialect=lwe)
mlir_tablegen(LWEOps.cpp.inc -gen-op-defs -dialect=lwe)
add_public_tablegen_target(MLIRLWEOpsIncGen)

set(LLVM_TARGET_DEFINITIONS LWETypes.td)
mlir_tablegen(LWETypes.h.inc -gen-typedef-decls -dialect=lwe)
mlir_tablegen(LWETypes.cpp.inc -gen-typedef-defs -dialect=lwe)
add_public_tablegen_target(MLIRLWETypesIncGen)


set(LLVM_TARGET_DEFINITIONS LWEAttributes.td)
mlir_tablegen(LWEAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect="lwe")
mlir_tablegen(LWEAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect="lwe")
add_public_tablegen_target(MLIRLWEAttributesIncGen)
