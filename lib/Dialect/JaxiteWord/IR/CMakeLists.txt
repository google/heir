add_heir_dialect(JaxiteWord jaxiteword)

add_mlir_dialect_library(HEIRJaxiteWord
    JaxiteWordDialect.cpp

    DEPENDS
    HEIRJaxiteWordIncGen
)
