add_mlir_library(HEIRRotationAnalysis
        RotationAnalysis.cpp

        DEPENDS

        LINK_LIBS PUBLIC
        HEIRTensorExt
        LLVMSupport
        MLIRAnalysis
        MLIRArithDialect
        MLIRIR
        MLIRSupport
)
target_link_libraries(HEIRAnalysis INTERFACE HEIRRotationAnalysis)
