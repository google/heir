add_heir_pass(ElementwiseToAffine)

add_mlir_library(HeirElementwiseToAffine
    ElementwiseToAffine.cpp

    DEPENDS
    HEIRElementwiseToAffineIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
)
target_link_libraries(HEIRTransforms INTERFACE HeirElementwiseToAffine)
