add_mlir_library(HEIRConversionUtils
    ConversionUtils.cpp

    # DEPENDS
    # HEIRTensorExtOpsIncGen
    # HEIRTensorExtPassesIncGen
    # HEIRTensorExtInsertRotateIncGen
    # HEIRTensorExtCanonicalizationIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
)

add_subdirectory(ModArithToArith)
add_subdirectory(BGVToOpenfhe)
add_subdirectory(BGVToLWE)
add_subdirectory(CGGIToJaxite)
add_subdirectory(CGGIToTfheRust)
add_subdirectory(CGGIToTfheRustBool)
add_subdirectory(SecretToCGGI)
add_subdirectory(MemrefToArith)
add_subdirectory(PolynomialToStandard)
add_subdirectory(SecretToBGV)
add_subdirectory(SecretToCKKS)
add_subdirectory(LWEToPolynomial)
add_subdirectory(LinalgToTensorExt)
add_subdirectory(TosaToSecretArith)
add_subdirectory(LWEToOpenfhe)
add_subdirectory(CKKSToOpenfhe)
add_subdirectory(RlweToOpenfhe)
