add_heir_dialect(ModArith mod_arith)

add_mlir_dialect_library(HEIRModArith
    ModArithDialect.cpp

    DEPENDS
    HEIRModArithIncGen
)
