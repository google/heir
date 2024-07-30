
add_mlir_dialect_library(MLIRMetadataEmitter
        MetadataEmitter.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRLWEOpsIncGen

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
