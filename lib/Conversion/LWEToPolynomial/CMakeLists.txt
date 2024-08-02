set(LLVM_TARGET_DEFINITIONS LWEToPolynomial.td)
mlir_tablegen(LWEToPolynomial.h.inc -gen-pass-decls -name LWEToPolynomial)
add_public_tablegen_target(MLIRLWEPolynomialTransformsIncGen)

add_mlir_dialect_library(MLIRLWEToPolynomial
    LWEToPolynomial.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRLWEPolynomialTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRLWE
    MLIRArithDialect

    MLIRIR
    MLIRPass
    MLIRPolynomialDialect

    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
