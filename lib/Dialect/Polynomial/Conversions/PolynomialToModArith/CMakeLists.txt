add_heir_pass(PolynomialToModArith)

add_mlir_conversion_library(HEIRPolynomialToModArith
    PolynomialToModArith.cpp

    DEPENDS
    HEIRPolynomialToModArithIncGen

    LINK_LIBS PUBLIC
    HEIRModArith
    HEIRConversionUtils

    LLVMSupport
    MLIRArithDialect
    MLIRDialectUtils
    MLIRIR
    MLIRLLVMDialect
    MLIRPass
    MLIRPolynomialDialect
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
