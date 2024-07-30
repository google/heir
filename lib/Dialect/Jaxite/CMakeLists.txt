add_subdirectory(IR)
add_mlir_dialect_library(MLIRJaxite
        IR/JaxiteDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRJaxiteIncGen
        MLIRJaxiteOpsIncGen
        MLIRJaxiteTypesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
