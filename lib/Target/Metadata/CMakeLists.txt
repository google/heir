
add_mlir_library(HEIRMetadataEmitter
        MetadataEmitter.cpp

        LINK_LIBS PUBLIC
        LLVMSupport
        MLIRAffineDialect
        MLIRArithDialect
        MLIRFuncDialect
        MLIRIR
        MLIRMemRefDialect
        MLIRSCFDialect
        MLIRSupport
        MLIRTranslateLib
)
target_link_libraries(HEIRTarget INTERFACE HEIRMetadataEmitter)
