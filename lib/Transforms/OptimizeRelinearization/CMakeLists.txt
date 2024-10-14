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
    MLIRSecret
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
)
