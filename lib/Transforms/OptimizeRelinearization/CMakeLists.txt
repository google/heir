add_heir_pass(OptimizeRelinearization)

add_mlir_library(HEIROptimizeRelinearization
    OptimizeRelinearization.cpp

    DEPENDS
    HEIROptimizeRelinearizationIncGen

    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    HEIROptimizeRelinearizationAnalysis
    MLIRIR
    MLIRPass
    HEIRSecret
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
)
target_link_libraries(HEIRTransforms INTERFACE HEIROptimizeRelinearization)
