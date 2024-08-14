set(LLVM_TARGET_DEFINITIONS ConvertIfToSelect.td)
mlir_tablegen(ConvertIfToSelect.h.inc -gen-pass-decls -name ConvertIfToSelect)
add_public_tablegen_target(MLIRHeirConvertIfToSelectIncGen)

add_mlir_dialect_library(MLIRHeirConvertIfToSelect
    ConvertIfToSelect.cpp

    DEPENDS
    MLIRHeirConvertIfToSelectIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
