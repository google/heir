add_heir_dialect(Secret secret)

add_mlir_dialect_library(HEIRSecret
    SecretDialect.cpp
    SecretOps.cpp
    SecretPatterns.cpp

    DEPENDS
    HEIRSecretIncGen
)
