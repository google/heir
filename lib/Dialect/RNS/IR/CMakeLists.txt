add_heir_dialect(RNS rns)

add_mlir_dialect_library(HEIRRNS
    RNSDialect.cpp
    RNSTypes.cpp
    RNSOps.cpp

    DEPENDS
    HEIRRNSIncGen
)
