add_heir_dialect(TfheRust tfhe_rust)

add_mlir_dialect_library(HEIRTfheRust
    TfheRustDialect.cpp
    TfheRustPatterns.cpp

    DEPENDS
    HEIRTfheRustIncGen
)
