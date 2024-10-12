add_heir_dialect(Openfhe openfhe)

add_mlir_dialect_library(HEIROpenfhe
    OpenfheDialect.cpp

    DEPENDS
    HEIROpenfheIncGen
)
