set(LLVM_TARGET_DEFINITIONS LinalgCanonicalizations.td)
mlir_tablegen(LinalgCanonicalizations.h.inc -gen-pass-decls -name LinalgCanonicalizations)
add_public_tablegen_target(MLIRHeirLinalgCanonicalizationsIncGen)

add_mlir_dialect_library(MLIRHeirLinalgCanonicalizations
    LinalgCanonicalizations.cpp

    DEPENDS
    MLIRHeirLinalgCanonicalizationsIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRSupport
    MLIRDialect
    MLIRIR
)
