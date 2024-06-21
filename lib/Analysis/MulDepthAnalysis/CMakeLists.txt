add_mlir_conversion_library(MLIRHEIRMulDepthAnalysis
        MulDepthAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

        DEPENDS
        MLIROpenfheOpsIncGen

        LINK_LIBS PUBLIC
        MLIROpenfhe
        LLVMSupport
        MLIRAnalysis
        MLIRIR
        MLIRSupport
)
