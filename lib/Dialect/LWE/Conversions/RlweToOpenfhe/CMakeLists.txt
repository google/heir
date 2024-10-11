add_mlir_dialect_library(MLIRRlweToOpenfheTransforms
    RlweToOpenfhe.cpp

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRSupport
        MLIRSupport
        MLIRTransformUtils
        MLIRHEIRUtils
        MLIRLWE
        MLIROpenfhe
        MLIRArithDialect
        MLIRFuncDialect
        MLIRIR
)
