add_heir_pass(PolynomialToModArith)

add_mlir_conversion_library(HEIRPolynomialToModArith
    PolynomialToModArith.cpp

    DEPENDS
    HEIRPolynomialToModArithIncGen

    LINK_LIBS PUBLIC
    HEIRConversionUtils

    LLVMSupport
    MLIRAffineDialect
    MLIRArithDialect
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRFuncTransforms
    MLIRIR
    MLIRLLVMDialect
    MLIRLinalgDialect
    MLIRPass
    MLIRPolynomialDialect
    MLIRSCFDialect
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
