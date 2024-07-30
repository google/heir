set(LLVM_TARGET_DEFINITIONS BGVToPolynomial.td)
mlir_tablegen(BGVToPolynomial.h.inc -gen-pass-decls -name BGVToPolynomial)
add_public_tablegen_target(MLIRBGVPolynomialTransformsIncGen)

add_mlir_dialect_library(MLIRBGVPolynomialTransforms
    BGVToPolynomial.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRBGVPolynomialTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRBGV
    MLIRLWE
    MLIRArithDialect

    MLIRIR
    MLIRPass
    MLIRPolynomialDialect

    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
