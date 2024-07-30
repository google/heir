set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Polynomial)
add_public_tablegen_target(MLIRPolynomialPassesIncGen)

set(LLVM_TARGET_DEFINITIONS NTTRewrites.td)
mlir_tablegen(NTTRewrites.cpp.inc -gen-rewriters -name Polynomial)
add_public_tablegen_target(MLIRPolynomialNTTRewritePassesIncGen)
