add_mlir_library(MLIRHEIRTargetUtils STATIC
        Utils.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR
        LINK_LIBS PUBLIC
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
add_subdirectory(OpenFhePke)
add_subdirectory(Jaxite)
add_subdirectory(Metadata)
add_subdirectory(TfheRust)
add_subdirectory(TfheRustBool)
add_subdirectory(Verilog)
