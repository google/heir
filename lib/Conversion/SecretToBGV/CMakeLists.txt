set(LLVM_TARGET_DEFINITIONS SecretToBGV.td)
mlir_tablegen(SecretToBGV.h.inc -gen-pass-decls -name SecretToBGV)
add_public_tablegen_target(MLIRSecretTransformsIncGen)

add_mlir_dialect_library(MLIRSecretTransforms
    SecretToBGV.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRSecretTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils

    MLIRBGV
    MLIRLWE
    MLIRSecret
    MLIRTensorExt
    LLVMSupport

    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRPolynomialDialect
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
