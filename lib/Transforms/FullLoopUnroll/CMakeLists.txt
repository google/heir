add_heir_pass(FullLoopUnroll)

add_mlir_library(HEIRFullLoopUnroll
    FullLoopUnroll.cpp

    DEPENDS
    HEIRFullLoopUnrollIncGen

    LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRSupport
    MLIRDialect
)
target_link_libraries(HEIRTransforms INTERFACE HEIRFullLoopUnroll)
