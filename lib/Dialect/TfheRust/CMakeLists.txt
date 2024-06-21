add_subdirectory(IR)
add_mlir_dialect_library(MLIRTfheRust
        IR/TfheRustDialect.cpp
        IR/TfheRustPatterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRTfheRustIncGen
        MLIRTfheRustOpsIncGen
        MLIRTfheRustTypesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
