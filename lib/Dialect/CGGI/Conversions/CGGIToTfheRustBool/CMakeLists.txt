set(LLVM_TARGET_DEFINITIONS CGGIToTfheRustBool.td)
mlir_tablegen(CGGIToTfheRustBool.h.inc -gen-pass-decls -name=CGGIToTfheRustBool)
add_public_tablegen_target(MLIRCGGIToTfheRustBoolTransformsIncGen)

add_mlir_dialect_library(MLIRCGGIToTfheRustBoolTransforms
    CGGIToTfheRustBool.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRCGGIToTfheRustBoolTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRCGGI
    MLIRLWE
    MLIRTfheRustBool
    LLVMSupport

    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass

    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
