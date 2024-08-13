add_subdirectory(IR)
add_mlir_dialect_library(MLIRModArithDialect
        IR/ModArithDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRModArithIncGen
        MLIRModArithOpsIncGen

        LINK_LIBS PUBLIC
        LLVMSupport
        MLIRArithDialect
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
)
