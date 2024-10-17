add_heir_pass(TensorToScalars)

add_mlir_library(HEIRTensorToScalars
    TensorToScalars.cpp

    DEPENDS
    HEIRTensorToScalarsIncGen

    LINK_LIBS PUBLIC
    HEIRSecret
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
)
target_link_libraries(HEIRTransforms INTERFACE HEIRTensorToScalars)
