set(LLVM_TARGET_DEFINITIONS ConvertSecretExtractToStaticExtract.td)
mlir_tablegen(ConvertSecretExtractToStaticExtract.h.inc -gen-pass-decls -name ConvertSecretExtractToStaticExtract)
add_public_tablegen_target(MLIRHeirConvertSecretExtractToStaticExtractIncGen)

add_mlir_dialect_library(MLIRHeirConvertSecretExtractToStaticExtract
    ConvertSecretExtractToStaticExtract.cpp

    DEPENDS
    MLIRHeirConvertSecretExtractToStaticExtractIncGen

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
