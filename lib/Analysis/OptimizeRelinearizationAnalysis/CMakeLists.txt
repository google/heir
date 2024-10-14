add_mlir_library(HEIROptimizeRelinearizationAnalysis
        OptimizeRelinearizationAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

        LINK_LIBS PUBLIC
        HEIRBGV
        LLVMSupport
        MLIRAnalysis
        MLIRIR
        MLIRSupport
        ortools::ortools
)
