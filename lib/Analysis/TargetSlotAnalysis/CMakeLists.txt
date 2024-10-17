add_mlir_library(HEIRTargetSlotAnalysis
        TargetSlotAnalysis.cpp

        LINK_LIBS PUBLIC
        LLVMSupport
        MLIRAnalysis
        MLIRSupport
        MLIRIR
        MLIRTensorDialect
        MLIRInferTypeOpInterface
)
target_link_libraries(HEIRAnalysis INTERFACE HEIRTargetSlotAnalysis)
