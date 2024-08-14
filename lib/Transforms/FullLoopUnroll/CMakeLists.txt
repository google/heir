set(LLVM_TARGET_DEFINITIONS FullLoopUnroll.td)
mlir_tablegen(FullLoopUnroll.h.inc -gen-pass-decls -name FullLoopUnroll)
add_public_tablegen_target(MLIRHeirFullLoopUnrollIncGen)

add_mlir_dialect_library(MLIRHeirFullLoopUnroll
    FullLoopUnroll.cpp

    DEPENDS
    MLIRHeirFullLoopUnrollIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRSupport
    MLIRDialect
    MLIRIR
)
