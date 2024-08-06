add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRCGGI
        IR/CGGIDialect.cpp
        IR/CGGIOps.cpp
        Transforms/SetDefaultParameters.cpp
    Transforms/BooleanLineVectorizer.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRCGGIIncGen
        MLIRCGGIOpsIncGen
        MLIRCGGIPassesIncGen
        MLIRCGGIAttributesIncGen

        LINK_LIBS PUBLIC
        MLIRHEIRInterfaces
        MLIRLWE
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
add_dependencies(MLIRCGGI MLIRCGGIAttributesIncGen)
