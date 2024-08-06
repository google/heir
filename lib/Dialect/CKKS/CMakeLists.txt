add_subdirectory(IR)
add_mlir_dialect_library(MLIRCKKS
        IR/CKKSDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRCKKSOpsIncGen
        MLIRCKKSIncGen

        LINK_LIBS PUBLIC
    MLIRLWE

    LLVMSupport

    MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
)
