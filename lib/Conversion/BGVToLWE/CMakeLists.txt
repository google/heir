set(LLVM_TARGET_DEFINITIONS BGVToLWE.td)
mlir_tablegen(BGVToLWE.h.inc -gen-pass-decls -name BGVToLWE)
add_public_tablegen_target(MLIRBGVLWETransformsIncGen)

add_mlir_dialect_library(MLIRBGVLWETransforms
    BGVToLWE.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRBGVLWETransformsIncGen

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
