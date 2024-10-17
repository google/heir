add_heir_pass(UnusedMemRef)

add_mlir_library(HEIRUnusedMemRef
    PARTIAL_SOURCES_INTENDED
    UnusedMemRef.cpp

    DEPENDS
    HEIRUnusedMemRefIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
)
target_link_libraries(HEIRTransforms INTERFACE HEIRUnusedMemRef)
