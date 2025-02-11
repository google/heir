add_mlir_library(HEIRJaxiteWordEmitter
    JaxiteWordEmitter.cpp

    LINK_LIBS PUBLIC
    HEIRSelectVariableNames
    HEIRJaxite
    HEIRLWE
    HEIRTargetUtils
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
target_link_libraries(HEIRTarget INTERFACE HEIRJaxiteWordEmitter)
