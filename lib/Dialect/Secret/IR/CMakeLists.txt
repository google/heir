set(LLVM_TARGET_DEFINITIONS SecretDialect.td)
mlir_tablegen(SecretDialect.h.inc -gen-dialect-decls -dialect=secret)
mlir_tablegen(SecretDialect.cpp.inc -gen-dialect-defs -dialect=secret)
add_public_tablegen_target(MLIRSecretIncGen)

set(LLVM_TARGET_DEFINITIONS SecretOps.td)
mlir_tablegen(SecretOps.h.inc -gen-op-decls)
mlir_tablegen(SecretOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRSecretOpsIncGen)


set(LLVM_TARGET_DEFINITIONS SecretTypes.td)
mlir_tablegen(SecretTypes.h.inc -gen-typedef-decls -dialect=lwe)
mlir_tablegen(SecretTypes.cpp.inc -gen-typedef-defs -dialect=lwe)
add_public_tablegen_target(MLIRSecretTypesIncGen)
