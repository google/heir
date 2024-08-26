set(LLVM_TARGET_DEFINITIONS ConvertSecretInsertToStaticInsert.td)
mlir_tablegen(ConvertSecretInsertToStaticInsert.h.inc -gen-pass-decls -name ConvertSecretInsertToStaticInsert)
add_public_tablegen_target(MLIRHeirConvertSecretInsertToStaticInsertIncGen)

add_mlir_dialect_library(MLIRHeirConvertSecretInsertToStaticInsert
    ConvertSecretInsertToStaticInsert.cpp

    DEPENDS
    MLIRHeirConvertSecretInsertToStaticInsertIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRSecretnessAnalysis
    LLVMSupport
    MLIRAffineDialect
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransformUtils
)
