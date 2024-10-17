add_heir_dialect(Jaxite jaxite)

add_mlir_dialect_library(HEIRJaxite
    JaxiteDialect.cpp

    DEPENDS
    HEIRJaxiteIncGen
)
