set(LLVM_TARGET_DEFINITIONS ConvertSecretForToStaticFor.td)
mlir_tablegen(ConvertSecretForToStaticFor.h.inc -gen-pass-decls -name ConvertSecretForToStaticFor)
add_public_tablegen_target(MLIRHeirConvertSecretForToStaticForIncGen)

add_mlir_dialect_library(MLIRHeirConvertSecretForToStaticFor
    ConvertSecretForToStaticFor.cpp

    DEPENDS
    MLIRHeirConvertSecretForToStaticForIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRSecretnessAnalysis
    LLVMSupport
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRDialect
    MLIRTransformUtils
)
