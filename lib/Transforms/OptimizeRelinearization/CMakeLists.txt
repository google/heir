set(LLVM_TARGET_DEFINITIONS OptimizeRelinearization.td)
mlir_tablegen(OptimizeRelinearization.h.inc -gen-pass-decls -name OptimizeRelinearization)
add_public_tablegen_target(MLIRHeirOptimizeRelinearizationIncGen)

add_mlir_dialect_library(MLIRHeirOptimizeRelinearization
    OptimizeRelinearization.cpp

    DEPENDS
    MLIRHeirOptimizeRelinearizationIncGen

    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    MLIRHeirOptimizeRelinearizationAnalysis
    MLIRIR
    MLIRPass
    MLIRSecret
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
)
