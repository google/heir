add_heir_pass(CGGIToTfheRust)

add_mlir_conversion_library(HEIRCGGIToTfheRust
    CGGIToTfheRust.cpp
    DEPENDS
    HEIRCGGIToTfheRustIncGen

    LINK_LIBS PUBLIC
    HEIRCGGI
    HEIRLWE
    HEIRTfheRust
    HEIRConversionUtils

    MLIRSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
