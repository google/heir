set(LLVM_TARGET_DEFINITIONS PolynomialToStandard.td)
mlir_tablegen(PolynomialToStandard.h.inc -gen-pass-decls -name PolynomialToStandard)
add_public_tablegen_target(MLIRPolynomialTransformsIncGen)

add_mlir_dialect_library(MLIRPolynomialTransforms
    PolynomialToStandard.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRPolynomialTransformsIncGen

    LINK_LIBS PUBLIC

    MLIRHEIRUtils
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
