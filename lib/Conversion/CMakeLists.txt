add_mlir_dialect_library(MLIRHEIRUtils
    PARTIAL_SOURCES_INTENDED
    Utils.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRTensorExtOpsIncGen
    MLIRTensorExtPassesIncGen
    MLIRTensorExtInsertRotateIncGen
    MLIRTensorExtCanonicalizationIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)

add_subdirectory(ModArithToArith)
add_subdirectory(BGVToOpenfhe)
add_subdirectory(BGVToLWE)
add_subdirectory(CGGIToJaxite)
add_subdirectory(CGGIToTfheRust)
add_subdirectory(CGGIToTfheRustBool)
add_subdirectory(CombToCGGI)
add_subdirectory(MemrefToArith)
add_subdirectory(PolynomialToStandard)
add_subdirectory(SecretToBGV)
add_subdirectory(LWEToPolynomial)
