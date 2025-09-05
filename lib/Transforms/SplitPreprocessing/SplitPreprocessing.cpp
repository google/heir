#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h"

#include <memory>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SPLITPREPROCESSING
#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h.inc"

using func::FuncOp;
using tensor_ext::AssignLayoutOp;
using tensor_ext::OriginalTypeAttr;

namespace {

struct SplitPreprocessingPass
    : impl::SplitPreprocessingBase<SplitPreprocessingPass> {
  using SplitPreprocessingBase::SplitPreprocessingBase;

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    MLIRContext* context = &getContext();
    OpBuilder builder(context);

    SmallVector<AssignLayoutOp> assignLayoutOps;

    funcOp.walk([&](AssignLayoutOp op) {
      if (isa<arith::ConstantOp>(op.getOperand().getDefiningOp())) {
        assignLayoutOps.push_back(op);
      }
    });

    if (assignLayoutOps.empty()) {
      return;
    }

    // Create the new function for the preprocessed part.
    SmallVector<Type, 4> newArgTypes;
    for (auto arg : funcOp.getArguments()) {
      newArgTypes.push_back(arg.getType());
    }
    for (auto op : assignLayoutOps) {
      newArgTypes.push_back(op.getResult().getType());
    }

    auto newFuncType =
        FunctionType::get(context, newArgTypes, funcOp.getResultTypes());
    auto newFuncName = funcOp.getName().str() + "__preprocessed";
    auto newFuncOp = FuncOp::create(funcOp.getLoc(), newFuncName, newFuncType);
    newFuncOp.setVisibility(funcOp.getVisibility());
    builder.setInsertionPointToEnd(newFuncOp.addEntryBlock());

    // Copy func arg attrs and result attrs to new func. This ensures the layout
    // attributes on the original function arguments are preserved in the new
    // preprocessed func.
    for (int i = 0; i < funcOp.getNumArguments(); ++i) {
      newFuncOp.setArgAttrs(i, funcOp.getArgAttrDict(i));
    }
    for (int i = 0; i < funcOp.getNumResults(); ++i) {
      newFuncOp.setResultAttrs(i, funcOp.getResultAttrDict(i));
    }

    IRMapping mapping;
    for (const auto& [index, value] : llvm::enumerate(funcOp.getArguments())) {
      BlockArgument arg = newFuncOp.getArgument(index);
      mapping.map(value, arg);
    }
    for (const auto& [index, op] : llvm::enumerate(assignLayoutOps)) {
      BlockArgument arg =
          newFuncOp.getArgument(funcOp.getNumArguments() + index);
      llvm::errs() << "Mapping " << op.getResult() << " to " << arg << "\n";
      mapping.map(op.getResult(), arg);
    }

    for (auto& op : funcOp.getBody().getOps()) {
      // Skip both the assign layout op and its feeder constant op
      Operation* opToCheck = &op;
      if (isa<arith::ConstantOp>(opToCheck) && opToCheck->hasOneUse()) {
        opToCheck = *op.getUsers().begin();
      }
      if (auto assignLayoutOp = dyn_cast<AssignLayoutOp>(opToCheck)) {
        if (llvm::find(assignLayoutOps, assignLayoutOp) !=
            assignLayoutOps.end()) {
          continue;
        }
      }
      builder.clone(op, mapping);
    }

    // Add original_type attribute to the new arguments.
    for (auto it : llvm::enumerate(assignLayoutOps)) {
      auto op = it.value();
      auto newArgIndex = funcOp.getNumArguments() + it.index();
      newFuncOp.setArgAttr(
          newArgIndex, tensor_ext::TensorExtDialect::kOriginalTypeAttrName,
          OriginalTypeAttr::get(context, op.getOperand().getType(),
                                op.getLayout()));
    }

    // Insert the new function before the original one.
    funcOp->getParentOfType<ModuleOp>().insert(funcOp.getOperation(),
                                               newFuncOp);

    // Replace the body of the original function with a call to the new one.
    Block* originalEntry = &funcOp.getBody().front();
    Operation* originalTerminator = originalEntry->getTerminator();
    builder.setInsertionPoint(originalTerminator);
    DenseSet<Operation*> opsToKeep = {originalTerminator};

    SmallVector<Value, 4> callOperands;
    for (auto arg : funcOp.getArguments()) {
      callOperands.push_back(arg);
    }
    for (auto op : assignLayoutOps) {
      callOperands.push_back(op.getResult());
      opsToKeep.insert(op);
      if (isa<arith::ConstantOp>(op.getOperand().getDefiningOp())) {
        opsToKeep.insert(op.getOperand().getDefiningOp());
      }
    }

    auto callOp =
        func::CallOp::create(builder, funcOp.getLoc(), newFuncOp, callOperands);
    opsToKeep.insert(callOp);
    originalTerminator->setOperands(callOp.getResults());

    for (Operation& op : llvm::make_early_inc_range(
             llvm::reverse(originalEntry->getOperations()))) {
      if (!opsToKeep.contains(&op)) op.erase();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createSplitPreprocessingPass() {
  return std::make_unique<SplitPreprocessingPass>();
}

}  // namespace heir
}  // namespace mlir
