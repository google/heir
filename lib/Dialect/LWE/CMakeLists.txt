add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRLWE
        IR/LWEDialect.cpp
        Transforms/SetDefaultParameters.cpp
        Transforms/AddClientInterface.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRLWEIncGen
        MLIRLWETypesIncGen
        MLIRLWEOpsIncGen
        MLIRLWEAttributesIncGen
        MLIRLWEPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
add_dependencies(MLIRLWE MLIRLWEIncGen)
add_dependencies(MLIRLWE MLIRLWEPassesIncGen)
