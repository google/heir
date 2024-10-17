add_heir_pass(CGGIToJaxite)

add_mlir_conversion_library(HEIRCGGIToJaxite
    CGGIToJaxite.cpp

    DEPENDS
    HEIRCGGIToJaxiteIncGen

    LINK_LIBS PUBLIC
    HEIRCGGI
    HEIRJaxite
    HEIRLWE
    HEIRConversionUtils

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
