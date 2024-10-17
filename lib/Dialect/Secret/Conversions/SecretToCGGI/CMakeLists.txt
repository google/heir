set(LLVM_TARGET_DEFINITIONS SecretToCGGI.td)
mlir_tablegen(SecretToCGGI.h.inc -gen-pass-decls -name SecretToCGGI)
add_public_tablegen_target(MLIRSecretTransformsIncGen)

add_mlir_dialect_library(MLIRSecretTransforms
    SecretToCGGI.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRSecretTransformsIncGen

    LINK_LIBS PUBLIC

    MLIRHEIRUtils
    MLIRCGGI
    MLIRComb
    MLIRLWE
    MLIRSecret

    LLVMSupport
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSupport
    MLIRTransformUtils
)
