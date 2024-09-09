set(LLVM_TARGET_DEFINITIONS SecretToCKKS.td)
mlir_tablegen(SecretToCKKS.h.inc -gen-pass-decls -name SecretToCKKS)
add_public_tablegen_target(MLIRSecretToCKKSTransformsIncGen)

add_mlir_dialect_library(MLIRSecretToCKKSTransforms
    SecretToCKKS.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRSecretToCKKSTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils

    MLIRCKKS
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
