add_heir_pass(ForwardStoreToLoad)

add_mlir_library(HEIRForwardStoreToLoad
    ForwardStoreToLoad.cpp

    DEPENDS
    HEIRForwardStoreToLoadIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
)
target_link_libraries(HEIRTransforms INTERFACE HEIRForwardStoreToLoad)
