set(LLVM_TARGET_DEFINITIONS CGGIToJaxite.td)
mlir_tablegen(CGGIToJaxite.h.inc -gen-pass-decls -name=CGGIToJaxite)
add_public_tablegen_target(MLIRCGGIToJaxiteTransformsIncGen)

add_mlir_dialect_library(MLIRCGGIToJaxiteTransforms
    CGGIToJaxite.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRCGGIToJaxiteTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRCGGI
    MLIRJaxite
    MLIRLWE

    LLVMSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
