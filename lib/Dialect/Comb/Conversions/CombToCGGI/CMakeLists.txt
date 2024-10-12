add_heir_pass(CombToCGGI)

add_mlir_conversion_library(HEIRCombToCGGI
    CombToCGGI.cpp

    DEPENDS
    HEIRCombToCGGIIncGen

    LINK_LIBS PUBLIC

    HEIRConversionUtils
    HEIRCGGI
    HEIRComb
    HEIRLWE
    HEIRSecret

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
