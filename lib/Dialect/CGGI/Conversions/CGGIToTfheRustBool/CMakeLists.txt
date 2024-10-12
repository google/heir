add_heir_pass(CGGIToTfheRustBool)

add_mlir_conversion_library(HEIRCGGIToTfheRustBool
    CGGIToTfheRustBool.cpp

    DEPENDS
    HEIRCGGIToTfheRustBoolIncGen

    LINK_LIBS PUBLIC
    HEIRCGGI
    HEIRLWE
    HEIRTfheRustBool
    HEIRConversionUtils

    LLVMSupport

    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRArithDialect
    MLIRTransformUtils
    MLIRTransforms
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
