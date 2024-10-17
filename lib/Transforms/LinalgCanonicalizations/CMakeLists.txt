add_heir_pass(LinalgCanonicalizations)

add_mlir_library(HEIRLinalgCanonicalizations
    LinalgCanonicalizations.cpp

    DEPENDS
    HEIRLinalgCanonicalizationsIncGen

    LINK_LIBS PUBLIC
    MLIRLinalgDialect
    MLIRInferTypeOpInterface
    MLIRSupport
    MLIRDialect
    MLIRIR
)
target_link_libraries(HEIRTransforms INTERFACE HEIRLinalgCanonicalizations)
