set(LLVM_TARGET_DEFINITIONS ModArithToArith.td)
mlir_tablegen(ModArithToArith.h.inc -gen-pass-decls -name ModArithToArith)
mlir_tablegen(ModArithToArith.cpp.inc -gen-rewriters -name ModArithToArith)
add_public_tablegen_target(MLIRModArithTransformsIncGen)

add_mlir_dialect_library(MLIRModArithTransforms
    ModArithToArith.cpp

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/IR

    DEPENDS
    MLIRModArithTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
)
