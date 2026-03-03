#ifndef LIB_ANALYSIS_ROTATIONANALYSIS_DAGBUILDER_H_
#define LIB_ANALYSIS_ROTATIONANALYSIS_DAGBUILDER_H_

#include <memory>

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

/// A helper to build an ArithmeticDag from MLIR for the purpose of simulating
/// the IR to extract rotation indices.
class DagBuilder {
  using Node = kernel::ArithmeticDagNode<kernel::LiteralValue>;
  using NodePtr = std::shared_ptr<Node>;

 public:
  DagBuilder() {}

  FailureOr<NodePtr> build(Operation* op);

 private:
  NodePtr findNodeOrMakeNewVariable(Value value);

  // Visit a block and return the NodePtr corresponding to its terminator
  FailureOr<NodePtr> visitBlockWithSingleTerminator(Block* block);

  FailureOr<NodePtr> visit(arith::AddIOp op);
  FailureOr<NodePtr> visit(arith::ConstantOp op);
  FailureOr<NodePtr> visit(arith::DivSIOp op);
  FailureOr<NodePtr> visit(arith::MulIOp op);
  FailureOr<NodePtr> visit(arith::SubIOp op);
  FailureOr<NodePtr> visit(scf::ForOp op);
  FailureOr<NodePtr> visit(scf::YieldOp op);
  FailureOr<NodePtr> visit(tensor::ExtractOp op);
  FailureOr<NodePtr> visit(tensor::SplatOp op);
  FailureOr<NodePtr> visit(tensor_ext::RotateOp op);

  // A mapping of previously visited Values
  DenseMap<Value, NodePtr> valueToNode;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ROTATIONANALYSIS_DAGBUILDER_H_
