set(LLVM_TARGET_DEFINITIONS ForwardStoreToLoad.td)
mlir_tablegen(ForwardStoreToLoad.h.inc -gen-pass-decls -name ForwardStoreToLoad)
add_public_tablegen_target(MLIRHeirForwardStoreToLoadIncGen)

add_mlir_dialect_library(MLIRHeirForwardStoreToLoad
    ForwardStoreToLoad.cpp

    DEPENDS
    MLIRHeirForwardStoreToLoadIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
