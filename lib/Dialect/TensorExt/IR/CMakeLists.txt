add_heir_dialect(TensorExt tensor_ext)

# Canonicalization patterns
set(LLVM_TARGET_DEFINITIONS TensorExtCanonicalization.td)
mlir_tablegen(TensorExtCanonicalization.cpp.inc -gen-rewriters)
add_public_tablegen_target(HEIRTensorExtCanonicalizationIncGen)

add_mlir_dialect_library(HEIRTensorExt
    TensorExtDialect.cpp
    TensorExtOps.cpp

    DEPENDS
    HEIRTensorExtIncGen
    HEIRTensorExtCanonicalizationIncGen
)
