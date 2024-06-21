
add_mlir_dialect_library(MLIROpenfheTarget
        OpenFhePkeEmitter.cpp
        OpenFhePkeHeaderEmitter.cpp
        OpenFheUtils.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIROpenfheOpsIncGen

        LINK_LIBS PUBLIC
        MLIROpenfhe
        MLIRIR
        MLIRInferTypeOpInterface
)
