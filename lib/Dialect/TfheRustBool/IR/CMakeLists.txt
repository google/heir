
add_heir_dialect(TfheRustBool tfhe_rust_bool)

add_mlir_dialect_library(HEIRTfheRustBool
    TfheRustBoolDialect.cpp

    DEPENDS
    HEIRTfheRustBoolIncGen
)
