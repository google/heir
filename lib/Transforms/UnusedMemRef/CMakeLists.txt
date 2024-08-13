set(LLVM_TARGET_DEFINITIONS UnusedMemRef.td)
mlir_tablegen(UnusedMemRef.h.inc -gen-pass-decls -name UnusedMemRef)
add_public_tablegen_target(MLIRHeirUnusedMemRefIncGen)

add_mlir_dialect_library(MLIRHeirUnusedMemRef
    PARTIAL_SOURCES_INTENDED
    UnusedMemRef.cpp

    DEPENDS
    MLIRHeirUnusedMemRefIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
