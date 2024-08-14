set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Secretize)
add_public_tablegen_target(MLIRHeirSecretizeIncGen)

add_mlir_dialect_library(MLIRHeirSecretize
    Secretize.cpp
    WrapGeneric.cpp

    DEPENDS
    MLIRHeirSecretizeIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
