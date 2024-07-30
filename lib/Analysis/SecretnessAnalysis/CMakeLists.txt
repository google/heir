add_mlir_conversion_library(MLIRHEIRSecretnessAnalysis
        SecretnessAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}

        DEPENDS
        MLIRSecretIncGen

        LINK_LIBS PUBLIC
        MLIRSecret
        LLVMSupport
        MLIRAnalysis
        MLIRSCFDialect
        MLIRIR
        MLIRSupport
)
