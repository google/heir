#include <cstddef>
#include <string>

#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"            // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"        // from @llvm-project

// ISL
#include "include/isl/ast.h"       // from @isl
#include "include/isl/ast_type.h"  // from @isl
#include "include/isl/ctx.h"       // from @isl

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

// A debugging helper to generate a C string representation of the loop nest.
FailureOr<std::string> generateLoopNestAsCStr(
    const presburger::IntegerRelation& rel);

// Generate an MLIR loop nest from an ISL AST.
class MLIRLoopNestGenerator {
 public:
  MLIRLoopNestGenerator(ImplicitLocOpBuilder& builder)
      : builder_(builder), ctx_(isl_ctx_alloc()) {}

  ~MLIRLoopNestGenerator() { isl_ctx_free(ctx_); }

  // Assumes that the tree is a perfect loop nest.
  FailureOr<scf::ForOp> generateForLoop(
      const presburger::IntegerRelation& rel, ValueRange initArgs,
      function_ref<scf::ValueVector(OpBuilder&, Location, ValueRange,
                                    ValueRange)>
          bodyBuilder);

 private:
  ImplicitLocOpBuilder& builder_;
  isl_ctx* ctx_;
};

}  // namespace heir
}  // namespace mlir
