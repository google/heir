add_heir_dialect(LWE lwe)

add_mlir_dialect_library(HEIRLWE
        LWEDialect.cpp

        DEPENDS
        HEIRLWEIncGen
)
