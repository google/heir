set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TensorExt)
add_public_tablegen_target(MLIRTensorExtPassesIncGen)

set(LLVM_TARGET_DEFINITIONS InsertRotate.td)
mlir_tablegen(InsertRotate.cpp.inc -gen-rewriters -name TensorExt)
add_public_tablegen_target(MLIRTensorExtInsertRotateIncGen)
