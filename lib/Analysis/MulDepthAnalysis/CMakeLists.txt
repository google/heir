add_mlir_library(HEIRMulDepthAnalysis
        MulDepthAnalysis.cpp

        LINK_LIBS PUBLIC
        HEIROpenfhe
        LLVMSupport
        MLIRAnalysis
        MLIRIR
        MLIRSupport
)
target_link_libraries(HEIRAnalysis INTERFACE HEIRMulDepthAnalysis)
