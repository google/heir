add_heir_pass(OperationBalancer)

add_mlir_library(HEIROperationBalancer
    OperationBalancer.cpp

    DEPENDS
    HEIROperationBalancerIncGen

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
target_link_libraries(HEIRTransforms INTERFACE HEIROperationBalancer)
