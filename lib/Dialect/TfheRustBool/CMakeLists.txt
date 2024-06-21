add_subdirectory(IR)
add_mlir_dialect_library(MLIRTfheRustBool
        IR/TfheRustBoolDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRTfheRustBoolIncGen
        MLIRTfheRustBoolOpsIncGen
        MLIRTfheRustBoolOpsTypesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
