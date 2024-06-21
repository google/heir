add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIROpenfhe
        IR/OpenfheDialect.cpp
        Transforms/ConfigureCryptoContext.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIROpenfheOpsIncGen
        MLIROpenfheTypesIncGen
        MLIROpenfheIncGen
        MLIROpenfhePassesIncGen

        LINK_LIBS PUBLIC
        LLVMSupport
        MLIRAnalysis
        MLIRArithDialect
        MLIRFuncDialect
        MLIRIR
        MLIRPass
        MLIRSupport
        MLIRInferTypeOpInterface
)

add_dependencies(MLIROpenfhe MLIRLWE)
