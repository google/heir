add_heir_pass(PolynomialToStandard)

add_mlir_conversion_library(HEIRPolynomialToStandard
    PolynomialToStandard.cpp

    DEPENDS
    HEIRPolynomialToStandardIncGen

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
