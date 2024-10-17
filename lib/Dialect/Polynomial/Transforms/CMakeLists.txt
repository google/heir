set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Polynomial)
add_public_tablegen_target(HEIRPolynomialPassesIncGen)

set(LLVM_TARGET_DEFINITIONS NTTRewrites.td)
mlir_tablegen(NTTRewrites.cpp.inc -gen-rewriters)
add_public_tablegen_target(HEIRPolynomialNTTRewritePassesIncGen)

add_mlir_library(HEIRPolynomialTransforms
    NTTRewrites.cpp

    DEPENDS
    HEIRPolynomialPassesIncGen
    HEIRPolynomialNTTRewritePassesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRDialect
)
