add_mlir_dialect_library(MLIRLWEOpenfheTransforms
    LWEToOpenfhe.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRLWE
    MLIRLWE
    MLIROpenfhe

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
