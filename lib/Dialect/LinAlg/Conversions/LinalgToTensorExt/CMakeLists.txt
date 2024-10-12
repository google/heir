add_heir_pass(LinalgToTensorExt)

add_mlir_conversion_library(HEIRLinalgToTensorExt
    LinalgToTensorExt.cpp

    DEPENDS
    HEIRLinalgToTensorExtIncGen

    LINK_LIBS PUBLIC
        HEIRSecretnessAnalysis
        HEIRTensorExt
        LLVMSupport
        MLIRAffineDialect
        MLIRAnalysis
        MLIRArithDialect
        MLIRIR
        MLIRLinalgDialect
        MLIRPass
        MLIRSupport
        MLIRTensorDialect
        MLIRTransformUtils
)
