#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
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

using BodyBuilderFn = function_ref<scf::ValueVector(OpBuilder&, Location,
                                                    ValueRange, ValueRange)>;

// Generate an ISL AST representing a loop nest from an IntegerRelation.
//
// This is intended to support the case where the IntegerRelation defines a
// single polyhedron representing a ciphertext layout, and the code generated
// for the packing can be expressed as a single perfect loop nest with a single
// assignment operator in the innermost loop body.
//
// domainIndicesToSchedule allows the caller to specify which domain dimensions
// should be part of the loop nest schedule. By default, only the range
// dimensions are scheduled. This determines which variables are used as loop
// indices.
FailureOr<isl_ast_node*> generateLoopNest(
    const presburger::IntegerRelation& rel, isl_ctx* ctx,
    ArrayRef<int> domainIndicesToSchedule = {});

// Calls `fn(domainPoint, rangePoint)` once for every point of `rel`, in
// lexicographic order of (domain, range). The points come from walking an isl
// AST built once for `rel`, so the isl work does not depend on the number of
// points. Every local of `rel` becomes a coordinate of that AST (a loop or an
// expression). A local that is not a floor division of the variables can take
// several values for one point; the walk then reaches the point once per
// value and reports it once. When `shouldStop` is provided and returns true,
// the scan stops early.
void forEachPoint(const presburger::IntegerRelation& rel,
                  function_ref<void(ArrayRef<int64_t> domainPoint,
                                    ArrayRef<int64_t> rangePoint)>
                      fn,
                  function_ref<bool()> shouldStop = {});

// A debugging helper to generate a C string representation of the loop nest.
FailureOr<std::string> generateLoopNestAsCStr(
    const presburger::IntegerRelation& rel,
    ArrayRef<int> domainIndicesToSchedule = {});

// Generate an MLIR loop nest from an ISL AST.
class MLIRLoopNestGenerator {
 public:
  MLIRLoopNestGenerator(ImplicitLocOpBuilder& builder)
      : builder_(builder),
        ctx_(isl_ctx_alloc()),
        currentLoc_(builder.getUnknownLoc()),
        createdOpCallback_([](Operation*) { return; }) {}

  MLIRLoopNestGenerator(
      ImplicitLocOpBuilder& builder,
      const std::function<void(Operation*)>& createdOpCallback)
      : builder_(builder),
        ctx_(isl_ctx_alloc()),
        currentLoc_(builder.getUnknownLoc()),
        createdOpCallback_(createdOpCallback) {}

  ~MLIRLoopNestGenerator() { isl_ctx_free(ctx_); }

  // Assumes that the tree is a perfect loop nest.
  FailureOr<scf::ForOp> generateForLoop(
      const presburger::IntegerRelation& rel, ValueRange initArgs,
      BodyBuilderFn bodyBuilder, ArrayRef<int> domainIndicesToSchedule = {},
      bool reverse = false);

 private:
  FailureOr<scf::ValueVector> visitAstNode(isl_ast_node* node,
                                           BodyBuilderFn bodyBuilder);
  FailureOr<scf::ValueVector> visitAstNodeFor(isl_ast_node* node,
                                              BodyBuilderFn bodyBuilder);
  FailureOr<scf::ValueVector> visitAstNodeIf(isl_ast_node* node,
                                             BodyBuilderFn bodyBuilder);
  FailureOr<scf::ValueVector> visitAstNodeBlock(isl_ast_node* node,
                                                BodyBuilderFn bodyBuilder);
  FailureOr<scf::ValueVector> visitAstNodeUser(isl_ast_node* node,
                                               BodyBuilderFn bodyBuilder);

  LogicalResult visitAstExpr(isl_ast_expr* expr);

  ImplicitLocOpBuilder& builder_;
  isl_ctx* ctx_;

  ValueRange currentIterArgs_;
  Location currentLoc_;
  std::map<std::string, Value> ivToValue_;
  SmallVector<scf::ForOp> loops_;

  const std::function<void(Operation*)> createdOpCallback_;
  bool reverse_ = false;
};

}  // namespace heir
}  // namespace mlir
