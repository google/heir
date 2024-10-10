set(LLVM_TARGET_DEFINITIONS ForwardInsertToExtract.td)
mlir_tablegen(ForwardInsertToExtract.h.inc -gen-pass-decls -name=ForwardInsertToExtract)

add_public_tablegen_target(MLIRHeirForwardInsertToExtractIncGen)

add_mlir_dialect_library(MLIRHeirForwardInsertToExtract
    ForwardInsertToExtract.cpp

    DEPENDS
    MLIRHeirForwardInsertToExtractIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
add_dependencies(MLIRHeirForwardInsertToExtract MLIRHeirForwardInsertToExtractIncGen)
