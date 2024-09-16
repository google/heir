set(LLVM_TARGET_DEFINITIONS BGVToOpenfhe.td)
mlir_tablegen(BGVToOpenfhe.h.inc -gen-pass-decls -name BGVToOpenfhe)
add_public_tablegen_target(MLIRBGVOpenfheTransformsIncGen)

add_mlir_dialect_library(MLIRBGVOpenfheTransforms
    BGVToOpenfhe.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRBGVOpenfheTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRBGV
    MLIRLWE
    MLIROpenfhe
    MLIRLWEOpenfheTransformsIncGen

    MLIRIR
    MLIRPass
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRFuncDialect
    LLVMSupport
    MLIRSupport
    MLIRDialect
    MLIRTransformUtils
    MLIRIR
)
