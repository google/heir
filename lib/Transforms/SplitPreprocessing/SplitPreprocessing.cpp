#include "lib/Transforms/SplitPreprocessing/SplitPreprocessing.h"

#include <memory>
#include <utility>

#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"             // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"                // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

#define DEBUG_TYPE "split-preprocessing"

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

    // Run secret canonicalization pattern to hoist tensor_ext.assign_layouts on
    // plaintexts outside of the secret.generic.
    mlir::RewritePatternSet patterns(context);
    patterns
        .add<secret::HoistPlaintextOps, secret::CollapseSecretlessGeneric,
             secret::ConcealThenGeneric, secret::RemoveNonSecretGenericArgs>(
            context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "failed to run secret canonicalization patterns";
      signalPassFailure();
      return;
    }

    SmallVector<Type, 4> newArgTypes;
    for (auto arg : funcOp.getArguments()) {
      newArgTypes.push_back(arg.getType());
    }

    DenseMap<AssignLayoutOp, int> assignLayoutToArgIndex;
    funcOp.walk([&](AssignLayoutOp op) {
      if (auto bbArg = dyn_cast<BlockArgument>(op.getOperand())) {
        assignLayoutToArgIndex[op] = bbArg.getArgNumber();
      } else {
        // Add new arguments for plaintext layout assignments on constants or
        // other values defined in the IR.
        newArgTypes.push_back(op.getResult().getType());
        assignLayoutToArgIndex[op] = newArgTypes.size() - 1;
      }
    });
    if (assignLayoutToArgIndex.empty()) {
      return;
    }

    auto newFuncType =
        FunctionType::get(context, newArgTypes, funcOp.getResultTypes());
    auto newFuncName = funcOp.getName().str() + "__preprocessed";
    auto newFuncOp = FuncOp::create(funcOp.getLoc(), newFuncName, newFuncType);
    newFuncOp.setVisibility(funcOp.getVisibility());

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
    Block* newEntryBlock = newFuncOp.addEntryBlock();
    mapping.map(&funcOp.front(), newEntryBlock);
    for (const auto& [index, value] : llvm::enumerate(funcOp.getArguments())) {
      mapping.map(value, newEntryBlock->getArgument(index));
    }

    for (auto& [op, index] : assignLayoutToArgIndex) {
      BlockArgument arg = newEntryBlock->getArgument(index);
      // Maps the results of the assign_layout to the new function arguments.
      LLVM_DEBUG(llvm::dbgs()
                     << "Mapping " << op.getResult() << " to " << arg << "\n";);
      mapping.map(op.getResult(), arg);
    }

    builder.setInsertionPointToEnd(newEntryBlock);
    for (auto& op : funcOp.getBody().getOps()) {
      if (auto assignLayoutOp = dyn_cast<AssignLayoutOp>(op)) continue;
      // If the mapping already contains all the ops results, then there's no
      // need to clone the op to the new function.
      if (!op.hasTrait<OpTrait::IsTerminator>() &&
          llvm::all_of(op.getResults(), [&](OpResult result) {
            return mapping.contains(result);
          })) {
        continue;
      }
      builder.clone(op, mapping);
    }

    // Add original_type attribute to the new arguments.
    for (auto& [op, index] : assignLayoutToArgIndex) {
      newFuncOp.setArgAttr(
          index, tensor_ext::TensorExtDialect::kOriginalTypeAttrName,
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

    SmallVector<Value> callOperands(newFuncOp.getNumArguments(), nullptr);
    for (auto arg : funcOp.getArguments()) {
      callOperands[arg.getArgNumber()] = arg;
    }
    for (auto& [op, index] : assignLayoutToArgIndex) {
      callOperands[index] = op.getResult();
    }

    auto callOp =
        func::CallOp::create(builder, funcOp.getLoc(), newFuncOp, callOperands);
    originalTerminator->setOperands(callOp.getResults());

    // Remove dead values in the new function operation - this isn't as simple
    // as removing all operations except for the assign layouts and their
    // operand's defining ops since there may be more complex plaintext
    // operations that are then used to define the operand.
    // Note: This doesn't remove any dead code from the newly created function,
    // since a dynamic pipeline must be scheduled under the root operation of
    // this pass (which is the original func::FuncOp).
    OpPassManager pipeline(func::FuncOp::getOperationName());
    pipeline.addPass(createRemoveDeadValuesPass());
    pipeline.addPass(createCSEPass());
    (void)runPipeline(pipeline, funcOp);
  }
};

}  // namespace

std::unique_ptr<Pass> createSplitPreprocessingPass() {
  return std::make_unique<SplitPreprocessingPass>();
}

}  // namespace heir
}  // namespace mlir
