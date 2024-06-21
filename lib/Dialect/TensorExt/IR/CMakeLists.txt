set(LLVM_TARGET_DEFINITIONS TensorExtDialect.td)
mlir_tablegen(TensorExtDialect.h.inc -gen-dialect-decls -dialect=tensor_ext)
mlir_tablegen(TensorExtDialect.cpp.inc -gen-dialect-defs -dialect=tensor_ext)
add_public_tablegen_target(MLIRTensorExtIncGen)

set(LLVM_TARGET_DEFINITIONS TensorExtOps.td)
mlir_tablegen(TensorExtOps.h.inc -gen-op-decls)
mlir_tablegen(TensorExtOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTensorExtOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TensorExtAttributes.td)
mlir_tablegen(TensorExtAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect="tensor_ext")
mlir_tablegen(TensorExtAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect="tensor_ext")
add_public_tablegen_target(MLIRTensorExtAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS TensorExtCanonicalization.td)
mlir_tablegen(TensorExtCanonicalization.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRTensorExtCanonicalizationIncGen)
