set(LLVM_TARGET_DEFINITIONS ConvertSecretWhileToStaticFor.td)
mlir_tablegen(ConvertSecretWhileToStaticFor.h.inc -gen-pass-decls -name ConvertSecretWhileToStaticFor)
add_public_tablegen_target(MLIRHeirConvertSecretWhileToStaticForIncGen)

add_mlir_dialect_library(MLIRHeirConvertSecretWhileToStaticFor
    ConvertSecretWhileToStaticFor.cpp

    DEPENDS
    MLIRHeirConvertSecretWhileToStaticForIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRSecretnessAnalysis
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransformUtils
)
