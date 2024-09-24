set(LLVM_TARGET_DEFINITIONS CKKSToOpenfhe.td)
mlir_tablegen(CKKSToOpenfhe.h.inc -gen-pass-decls -name CKKSToOpenfhe)
add_public_tablegen_target(MLIRCKKSOpenfheTransformsIncGen)

add_mlir_dialect_library(MLIRCKKSOpenfheTransforms
    CKKSToOpenfhe.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRCKKSOpenfheTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRCKKS
    MLIRLWE
    MLIROpenfhe
    MLIRLWEOpenfheTransforms
    MLIRRlweToOpenfheTransforms

    MLIRIR
    MLIRPass
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRFuncDialect
    LLVMSupport
    MLIRSupport
    MLIRDialect
    MLIRTransformUtils
    MLIRTensorDialect
)
