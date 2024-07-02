add_mlir_conversion_library(MLIRHeirOptimizeRelinearizationAnalysis
        OptimizeRelinearizationAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

        LINK_LIBS PUBLIC
        MLIRBGV
        LLVMSupport
        MLIRAnalysis
        MLIRIR
        MLIRSupport
        ortools::ortools
)
