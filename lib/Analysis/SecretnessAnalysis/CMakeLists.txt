add_mlir_library(HEIRSecretnessAnalysis
        SecretnessAnalysis.cpp

        LINK_LIBS PUBLIC
        HEIRSecret
        LLVMSupport
        MLIRAnalysis
        MLIRSCFDialect
        MLIRIR
        MLIRSupport
)
target_link_libraries(HEIRAnalysis INTERFACE HEIRSecretnessAnalysis)
