add_subdirectory(IR)
add_mlir_dialect_library(MLIRArithExtDialect
        IR/ArithExtDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRArithExtIncGen
        MLIRArithExtOpsIncGen

        LINK_LIBS PUBLIC
        LLVMSupport
        MLIRArithDialect
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
)
