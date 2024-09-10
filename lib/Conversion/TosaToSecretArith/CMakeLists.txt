set(LLVM_TARGET_DEFINITIONS TosaToSecretArith.td)
mlir_tablegen(TosaToSecretArith.h.inc -gen-pass-decls -name TosaToSecretArith)
add_public_tablegen_target(MLIRTosaToSecretArithTransformsIncGen)

add_mlir_dialect_library(MLIRTosaToSecretArith
    TosaToSecretArith.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRTosaToSecretArithTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRSecretnessAnalysis
    MLIRTensorExt
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTosaDialect
    MLIRTransformUtils
    LLVMSupport
    MLIRSupport
)
