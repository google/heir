set(LLVM_TARGET_DEFINITIONS ElementwiseToAffine.td)
mlir_tablegen(ElementwiseToAffine.h.inc -gen-pass-decls -name ElementwiseToAffine)
add_public_tablegen_target(MLIRHeirElementwiseToAffineIncGen)

add_mlir_dialect_library(MLIRHeiElementwiseToAffine
    ElementwiseToAffine.cpp

    DEPENDS
    MLIRHeirElementwiseToAffineIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
