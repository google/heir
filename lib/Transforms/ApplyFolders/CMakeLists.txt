set(LLVM_TARGET_DEFINITIONS ApplyFolders.td)
mlir_tablegen(ApplyFolders.h.inc -gen-pass-decls -name ApplyFolders)
add_public_tablegen_target(MLIRHeirApplyFoldersIncGen)

add_mlir_dialect_library(MLIRHeirApplyFolders
    ApplyFolders.cpp

    DEPENDS
    MLIRHeirApplyFoldersIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
