#include <cstdint>
#include <iterator>
#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulResultAnalysis/MulResultAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"
#include "lib/Dialect/Mgmt/Transforms/Passes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SECRETINSERTMGMTCKKS
#include "lib/Transforms/SecretInsertMgmt/Passes.h.inc"

namespace {

// Returns the unique non-unit dimension of a tensor and its rank.
// Returns failure if the tensor has more than one non-unit dimension.
FailureOr<std::pair<unsigned, int64_t>> getNonUnitDimension(
    RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();

  if (llvm::count_if(shape, [](auto dim) { return dim != 1; }) != 1) {
    return failure();
  }

  unsigned nonUnitIndex = std::distance(
      shape.begin(), llvm::find_if(shape, [&](auto dim) { return dim != 1; }));

  return std::make_pair(nonUnitIndex, shape[nonUnitIndex]);
}

bool isTensorInSlots(Operation *top, DataFlowSolver *solver, int slotNumber) {
  // Ensure that all secret types are uniform and matching the ring
  // parameter size in order to pack tensors into ciphertext SIMD slots.
  LogicalResult result = walkAndValidateValues(
      top,
      [&](Value value) {
        auto secret = isSecret(value, solver);
        if (secret) {
          auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
          if (tensorTy) {
            // TODO(#913): Multidimensional tensors with a single non-unit
            // dimension are assumed to be packed in the order of that
            // dimensions.
            auto nonUnitDim = getNonUnitDimension(tensorTy);
            if (failed(nonUnitDim) || nonUnitDim.value().second != slotNumber) {
              return failure();
            }
          }
        }
        return success();
      },
      "expected secret types to be tensors with dimension matching "
      "ring parameter, pass will not pack tensors into ciphertext "
      "SIMD slots");

  return succeeded(result);
}

void annotateTensorExtractAsNotSlotExtract(Operation *top,
                                           DataFlowSolver *solver) {
  top->walk([&](tensor::ExtractOp extractOp) {
    if (isSecret(extractOp.getOperand(0), solver)) {
      extractOp->setAttr("slot_extract",
                         BoolAttr::get(extractOp.getContext(), false));
    }
  });
}

}  // namespace

struct SecretInsertMgmtCKKS
    : impl::SecretInsertMgmtCKKSBase<SecretInsertMgmtCKKS> {
  using SecretInsertMgmtCKKSBase::SecretInsertMgmtCKKSBase;

  void runOnOperation() override {
    // Helper for future lowerings that want to know what scheme was used
    moduleSetCKKS(getOperation());

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // TODO(#1174): decide packing earlier in the pipeline instead of annotation
    // determine whether tensor::Extract is extracting slot from ciphertext
    // or generic tensor extract from tensor ciphertext
    // This is directly copied from secret-to-ckks
    // should merge into earlier pipeline
    bool packTensorInSlots =
        isTensorInSlots(getOperation(), &solver, slotNumber);
    if (!packTensorInSlots) {
      annotateTensorExtractAsNotSlotExtract(getOperation(), &solver);
    }

    // re-run analysis as MulResultAnalysis is affected by slot_extract
    solver.load<MulResultAnalysis>();
    solver.eraseAllStates();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    RewritePatternSet patternsRelinearize(&getContext());
    patternsRelinearize.add<MultRelinearize<arith::MulIOp>>(
        &getContext(), getOperation(), &solver);
    patternsRelinearize.add<MultRelinearize<arith::MulFOp>>(
        &getContext(), getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsRelinearize));

    RewritePatternSet patternsMultModReduce(&getContext());
    patternsMultModReduce.add<ModReduceBefore<arith::MulIOp>>(
        &getContext(), /*isMul*/ true, includeFirstMul, getOperation(),
        &solver);
    patternsMultModReduce.add<ModReduceBefore<arith::MulFOp>>(
        &getContext(), /*isMul*/ true, includeFirstMul, getOperation(),
        &solver);
    // tensor::ExtractOp = mulConst + rotate
    patternsMultModReduce.add<ModReduceBefore<tensor::ExtractOp>>(
        &getContext(), /*isMul*/ true, includeFirstMul, getOperation(),
        &solver);
    // isMul = true and includeFirstMul = false here
    // as before yield we want mulResult to be mod reduced
    patternsMultModReduce.add<ModReduceBefore<secret::YieldOp>>(
        &getContext(), /*isMul*/ true, /*includeFirstMul*/ false,
        getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(),
                               std::move(patternsMultModReduce));

    // insert BootstrapOp after mgmt::ModReduceOp
    // This must be run before level mismatch
    // NOTE: actually bootstrap before mod reduce is better
    // as after modreduce to level `0` there still might be add/sub
    // and these op done there could be minimal cost.
    // However, this greedy strategy is temporary so not too much
    // optimization now
    RewritePatternSet patternsBootstrapWaterLine(&getContext());
    patternsBootstrapWaterLine.add<BootstrapWaterLine<mgmt::ModReduceOp>>(
        &getContext(), getOperation(), &solver, bootstrapWaterline);
    (void)walkAndApplyPatterns(getOperation(),
                               std::move(patternsBootstrapWaterLine));

    // when other binary op operands level mismatch
    // includeFirstMul not used for these ops
    RewritePatternSet patternsAddModReduce(&getContext());
    patternsAddModReduce.add<ModReduceBefore<arith::AddIOp>>(
        &getContext(), /*isMul*/ false, /*includeFirstMul*/ false,
        getOperation(), &solver);
    patternsAddModReduce.add<ModReduceBefore<arith::AddFOp>>(
        &getContext(), /*isMul*/ false, /*includeFirstMul*/ false,
        getOperation(), &solver);
    patternsAddModReduce.add<ModReduceBefore<arith::SubIOp>>(
        &getContext(), /*isMul*/ false, /*includeFirstMul*/ false,
        getOperation(), &solver);
    patternsAddModReduce.add<ModReduceBefore<arith::SubFOp>>(
        &getContext(), /*isMul*/ false, /*includeFirstMul*/ false,
        getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsAddModReduce));

    // call CSE here because there may be redundant mod reduce
    // one Value may get mod reduced multiple times in
    // multiple Uses
    //
    // also run annotate-mgmt for lowering
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCSEPass());
    pipeline.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
