add_heir_pass(SecretToCGGI)

add_mlir_dialect_library(HEIRSecretToCGGI
    SecretToCGGI.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    HEIRSecretToCGGIIncGen

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
