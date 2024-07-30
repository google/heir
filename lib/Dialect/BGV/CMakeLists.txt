add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRBGV
        IR/BGVDialect.cpp
        Transforms/AddClientInterface.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRBGVIncGen
        MLIRBGVOpsIncGen
        MLIRBGVPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
