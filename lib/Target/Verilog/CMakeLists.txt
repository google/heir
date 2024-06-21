
add_mlir_dialect_library(MLIRVerilogTarget
        VerilogEmitter.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRSecretOpsIncGen

    LINK_LIBS PUBLIC
    MLIRMemrefToArithUtils
    MLIRSecret
    MLIRHEIRTargetUtils
    LLVMSupport
    MLIRTranslateLib
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRFuncDialect
    MLIRSupport
    MLIRIR
    MLIRInferTypeOpInterface

)
