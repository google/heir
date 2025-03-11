add_mlir_conversion_library(HEIRLWEToJaxiteWord
    LWEToJaxiteWord.cpp

    LINK_LIBS PUBLIC
    HEIRConversionUtils
    HEIRLWE
    HEIRLWE
    HEIRJaxiteWord

    MLIRIR
    MLIRPass
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRFuncDialect
    LLVMSupport
    MLIRSupport
    MLIRDialect
    MLIRTransformUtils
    MLIRIR
)
