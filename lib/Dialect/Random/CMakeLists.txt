add_subdirectory(IR)
add_mlir_dialect_library(MLIRRandom
        IR/RandomDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRRandomOpsIncGen
        MLIRRandomIncGen
        MLIRRandomTypesIncGen
        MLIRRandomEnumsIncGen

        LINK_LIBS PUBLIC
    LLVMSupport
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
)
