add_heir_pass(TosaToSecretArith)

add_mlir_library(HEIRTosaToSecretArith
    TosaToSecretArith.cpp

    DEPENDS
    HEIRTosaToSecretArithIncGen

    LINK_LIBS PUBLIC
    HEIRSecretnessAnalysis
    HEIRTensorExt
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTosaDialect
    MLIRTransformUtils
    LLVMSupport
    MLIRSupport
)
