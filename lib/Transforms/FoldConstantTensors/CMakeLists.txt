add_heir_pass(FoldConstantTensors)

add_mlir_library(HEIRFoldConstantTensors
    FoldConstantTensors.cpp

    DEPENDS
    HEIRFoldConstantTensorsIncGen

    LINK_LIBS PUBLIC
    MLIRTensorDialect
    MLIRInferTypeOpInterface
    MLIRSupport
    MLIRDialect
    MLIRIR
)
target_link_libraries(HEIRTransforms INTERFACE HEIRFoldConstantTensors)
