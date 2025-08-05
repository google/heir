#include "isl/ast_type.h"                                           // from @isl
#include "isl/ctx.h"                                                // from @isl
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

// Generate an ISL AST representing a loop nest from an IntegerRelation.
//
// This is intended to support the case where the IntegerRelation defines a
// single polyhedron representing a ciphertext layout, and the code generated
// for the packing can be expressed as a single perfect loop nest with a single
// assignment operator in the innermost loop body.
FailureOr<isl_ast_node*> generateLoopNest(
    const presburger::IntegerRelation& rel, isl_ctx* ctx);

}  // namespace heir
}  // namespace mlir
