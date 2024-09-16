set(LLVM_TARGET_DEFINITIONS LWEToOpenfhe.td)
mlir_tablegen(LWEToOpenfhe.h.inc -gen-pass-decls -name LWEToOpenfhe)
add_public_tablegen_target(MLIRLWEOpenfheTransformsIncGen)

add_mlir_dialect_library(MLIRLWEOpenfheTransforms
    LWEToOpenfhe.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRLWEOpenfheTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRLWE
    MLIRLWE
    MLIROpenfhe

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
