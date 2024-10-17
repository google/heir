add_mlir_conversion_library(HEIRRlweToOpenfhe
    RlweToOpenfhe.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRSupport
        MLIRTransformUtils
        HEIRConversionUtils
        HEIRLWE
        HEIROpenfhe
        MLIRArithDialect
        MLIRFuncDialect
        MLIRIR
)
