
add_mlir_dialect_library(MLIRJaxiteTarget
        JaxiteEmitter.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRJaxiteOpsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRAnalysisSVN
    MLIRJaxite
    MLIRLWE
    MLIRHEIRTargetUtils
    LLVMSupport
    MLIRArithDialect
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRSupport
    MLIRTensorDialect
    MLIRTranslateLib
)
