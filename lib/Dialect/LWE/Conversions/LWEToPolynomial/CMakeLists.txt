add_heir_pass(LWEToPolynomial)

add_mlir_conversion_library(HEIRLWEToPolynomial
    LWEToPolynomial.cpp

    DEPENDS
    HEIRLWEToPolynomialIncGen

    LINK_LIBS PUBLIC
    HEIRConversionUtils
    HEIRLWE
    HEIRRandom
    MLIRArithDialect

    MLIRIR
    MLIRPass
    MLIRPolynomialDialect

    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
