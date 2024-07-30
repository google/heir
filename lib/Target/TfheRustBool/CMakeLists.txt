
add_mlir_dialect_library(MLIRTfheRustBoolTarget
        TfheRustBoolEmitter.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRTfheRustBoolOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTfheRust
        MLIRIR
        MLIRInferTypeOpInterface
)
