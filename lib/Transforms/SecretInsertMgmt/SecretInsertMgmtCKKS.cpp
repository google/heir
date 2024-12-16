#include <cstdint>
#include <iterator>
#include <utility>

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulResultAnalysis/MulResultAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Transforms/SecretInsertMgmt/Passes.h"
#include "lib/Transforms/SecretInsertMgmt/SecretInsertMgmtPatterns.h"
#include "llvm/include/llvm/ADT/STLExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Iterators.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
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
  bool packTensorInSlots = true;
  WalkResult compatibleTensors = top->walk([&](Operation *op) {
    for (auto value : op->getOperands()) {
      auto secretness =
          solver->lookupState<SecretnessLattice>(value)->getValue();
      if (secretness.isInitialized() && secretness.getSecretness()) {
        auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
        if (tensorTy) {
          // TODO(#913): Multidimensional tensors with a single non-unit
          // dimension are assumed to be packed in the order of that
          // dimensions.
          auto nonUnitDim = getNonUnitDimension(tensorTy);
          if (failed(nonUnitDim)) {
            return WalkResult::interrupt();
          }
          if (nonUnitDim.value().second != slotNumber) {
            return WalkResult::interrupt();
          }
        }
      }
    }
    return WalkResult::advance();
  });
  if (compatibleTensors.wasInterrupted()) {
    emitWarning(top->getLoc(),
                "expected secret types to be tensors with dimension matching "
                "ring parameter, pass will not pack tensors into ciphertext "
                "SIMD slots");
    packTensorInSlots = false;
  }
  return packTensorInSlots;
}

void annotateTensorExtractAsNotSlotExtract(Operation *top,
                                           DataFlowSolver *solver) {
  top->walk([&](tensor::ExtractOp extractOp) {
    auto secretness =
        solver->lookupState<SecretnessLattice>(extractOp.getOperand(0))
            ->getValue();
    if (secretness.isInitialized() && secretness.getSecretness()) {
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

    // only after relinearize we can get the correct dimension
    // NOTE: for lazy relinearize, need to uninitialize the DimensionLattice...
    // otherwise DimensionState::join may not work correctly
    // if we are reusing a solver
    solver.load<DimensionAnalysis>();

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
    OpPassManager csePipeline("builtin.module");
    csePipeline.addPass(createCSEPass());
    (void)runPipeline(csePipeline, getOperation());

    // re-run analysis after CSE
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // annotate level and dimension from analysis
    annotateLevel(getOperation(), &solver);
    annotateDimension(getOperation(), &solver);
    // combine level and dimension into MgmtAttr
    // also removes the level/dimension annotations
    annotateMgmtAttr(getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
