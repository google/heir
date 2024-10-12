add_heir_dialect(Random random)

add_mlir_dialect_library(HEIRRandom
    RandomDialect.cpp

    DEPENDS
    HEIRRandomIncGen
)
