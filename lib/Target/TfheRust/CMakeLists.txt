
add_mlir_dialect_library(MLIRTfheRustTarget
        TfheRustEmitter.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRTfheRustOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTfheRust
        MLIRIR
        MLIRInferTypeOpInterface
)
