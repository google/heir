set(LLVM_TARGET_DEFINITIONS StraightLineVectorizer.td)
mlir_tablegen(StraightLineVectorizer.h.inc -gen-pass-decls -name StraightLineVectorizer)
add_public_tablegen_target(MLIRHeirStraightLineVectorizerIncGen)

add_mlir_dialect_library(MLIRHeirStraightLineVectorizer
    StraightLineVectorizer.cpp

    DEPENDS
    MLIRHeirStraightLineVectorizerIncGen

    LINK_LIBS PUBLIC
    MLIRModArithDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
