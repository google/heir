set(LLVM_TARGET_DEFINITIONS PolynomialToModArith.td)
mlir_tablegen(PolynomialToModArith.h.inc -gen-pass-decls -name PolynomialToModArith)
add_public_tablegen_target(MLIRPolynomialTransformsIncGen)

add_mlir_dialect_library(MLIRPolynomialTransforms
    PolynomialToModArith.cpp

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
