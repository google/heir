add_mlir_conversion_library(HEIRLWEToOpenfhe
    LWEToOpenfhe.cpp

    LINK_LIBS PUBLIC
    HEIRConversionUtils
    HEIRLWE
    HEIRLWE
    HEIROpenfhe

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
