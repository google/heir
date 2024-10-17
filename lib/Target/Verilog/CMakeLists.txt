
add_mlir_library(MLIRVerilogTarget
    VerilogEmitter.cpp

    LINK_LIBS PUBLIC
    HEIRMemrefToArithUtils
    HEIRSecret
    HEIRTargetUtils
    LLVMSupport
    MLIRTranslateLib
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRFuncDialect
    MLIRSupport
    MLIRIR
    MLIRInferTypeOpInterface
)
target_link_libraries(HEIRTarget INTERFACE MLIRVerilogTarget)
