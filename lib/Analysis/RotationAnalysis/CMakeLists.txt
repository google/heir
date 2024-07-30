add_mlir_conversion_library(MLIRHEIRAnalysisROT
        RotationAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

        DEPENDS
        MLIRTensorExtOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTensorExt
        LLVMSupport
        MLIRAnalysis
        MLIRArithDialect
        MLIRIR
        MLIRSupport
)
