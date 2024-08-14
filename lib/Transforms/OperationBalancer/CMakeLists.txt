set(LLVM_TARGET_DEFINITIONS OperationBalancer.td)
mlir_tablegen(OperationBalancer.h.inc -gen-pass-decls -name OperationBalancer)
add_public_tablegen_target(MLIRHeirOperationBalancerIncGen)

add_mlir_dialect_library(MLIRHeirOperationBalancer
    OperationBalancer.cpp

    DEPENDS
    MLIRHeirOperationBalancerIncGen

    LINK_LIBS PUBLIC
    MLIRSecret
    LLVMSupport
    MLIRAnalysis
    MLIRArithDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRTransforms
)
