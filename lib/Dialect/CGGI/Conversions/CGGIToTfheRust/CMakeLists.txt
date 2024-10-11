set(LLVM_TARGET_DEFINITIONS CGGIToTfheRust.td)
mlir_tablegen(CGGIToTfheRust.h.inc -gen-pass-decls -name CGGIToTfheRust)
add_public_tablegen_target(MLIRCGGIToTfheRustTransformsIncGen)

add_mlir_dialect_library(MLIRCGGIToTfheRustTransforms
    CGGIToTfheRust.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRCGGIToTfheRustTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRHEIRUtils
    MLIRCGGI
    MLIRLWE
    MLIRTfheRust

    MLIRSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSupport
    MLIRTensorDialect
    MLIRTransformUtils
)
