#include <cstdint>
#include <iterator>
#include <utility>

#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/MulDepthAnalysis/MulDepthAnalysis.h"
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
    // for Openfhe, use B/FV style mgmt: only relinearize, no level management.
    // still maintain the maximal level information though for lowering.
    if (moduleIsOpenfhe(getOperation())) {
      OpPassManager pipeline("builtin.module");
      pipeline.addPass(createSecretInsertMgmtBFV());
      (void)runPipeline(pipeline, getOperation());
      moduleSetCKKS(getOperation());
      return;
    }

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

    // handle plaintext operands
    RewritePatternSet patternsPlaintext(&getContext());
    patternsPlaintext.add<UseInitOpForPlaintextOperand<arith::AddIOp>,
                          UseInitOpForPlaintextOperand<arith::SubIOp>,
                          UseInitOpForPlaintextOperand<arith::MulIOp>,
                          UseInitOpForPlaintextOperand<arith::AddFOp>,
                          UseInitOpForPlaintextOperand<arith::SubFOp>,
                          UseInitOpForPlaintextOperand<arith::MulFOp>>(
        &getContext(), getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsPlaintext));

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

    // re-run analysis as MulDepthAnalysis is affected by slot_extract
    solver.load<MulDepthAnalysis>();
    solver.eraseAllStates();
    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    if (afterMul) {
      RewritePatternSet patternsMultModReduce(&getContext());
      patternsMultModReduce.add<ModReduceAfterMult<arith::MulIOp>,
                                ModReduceAfterMult<arith::MulFOp>,
                                ModReduceAfterMult<tensor::ExtractOp>>(
          &getContext(), getOperation(), &solver);
      (void)walkAndApplyPatterns(getOperation(),
                                 std::move(patternsMultModReduce));
    } else {
      RewritePatternSet patternsMultModReduce(&getContext());
      // tensor::ExtractOp = mulConst + rotate
      patternsMultModReduce
          .add<ModReduceBefore<arith::MulIOp>, ModReduceBefore<arith::MulFOp>,
               ModReduceBefore<tensor::ExtractOp>>(
              &getContext(), beforeMulIncludeFirstMul, getOperation(), &solver);
      // includeFirstMul = false here
      // as before yield we only want mulResult to be mod reduced
      patternsMultModReduce.add<ModReduceBefore<secret::YieldOp>>(
          &getContext(), /*includeFirstMul*/ false, getOperation(), &solver);
      (void)walkAndApplyPatterns(getOperation(),
                                 std::move(patternsMultModReduce));
    }

    // this must be run after ModReduceAfterMult
    RewritePatternSet patternsRelinearize(&getContext());
    patternsRelinearize
        .add<MultRelinearize<arith::MulIOp>, MultRelinearize<arith::MulFOp>>(
            &getContext(), getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsRelinearize));

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
    //
    // See also MatchCrossLevel documentation
    int idCounter = 0;  // for making adjust_scale op different to avoid cse
    RewritePatternSet patternsAddModReduce(&getContext());
    patternsAddModReduce
        .add<MatchCrossLevel<arith::AddIOp>, MatchCrossLevel<arith::SubIOp>,
             MatchCrossLevel<arith::MulIOp>, MatchCrossLevel<arith::AddFOp>,
             MatchCrossLevel<arith::SubFOp>, MatchCrossLevel<arith::MulFOp>>(
            &getContext(), &idCounter, getOperation(), &solver);
    (void)walkAndApplyPatterns(getOperation(), std::move(patternsAddModReduce));

    // when other binary op operands mulDepth mismatch
    // this only happen for before-mul but not include-first-mul case
    // at the first level, a Value can be both mulResult or not mulResult
    // we should match their scale by adding one adjust scale op
    //
    // See also MatchCrossMulDepth documentation
    if (!beforeMulIncludeFirstMul && !afterMul) {
      RewritePatternSet patternsMulDepth(&getContext());
      patternsMulDepth.add<
          MatchCrossMulDepth<arith::MulIOp>, MatchCrossMulDepth<arith::AddIOp>,
          MatchCrossMulDepth<arith::SubIOp>, MatchCrossMulDepth<arith::MulFOp>,
          MatchCrossMulDepth<arith::AddFOp>, MatchCrossMulDepth<arith::SubFOp>>(
          &getContext(), &idCounter, getOperation(), &solver);
      (void)walkAndApplyPatterns(getOperation(), std::move(patternsMulDepth));
    }

    // call Canonicalizer here because mgmt ops need to be ordered
    // call CSE here because there may be redundant mod reduce
    // one Value may get mod reduced multiple times in
    // multiple Uses
    //
    // also run annotate-mgmt for lowering
    OpPassManager pipeline("builtin.module");
    pipeline.addPass(createCanonicalizerPass());
    pipeline.addPass(createCSEPass());
    pipeline.addPass(mgmt::createAnnotateMgmt());
    (void)runPipeline(pipeline, getOperation());
  }
};

}  // namespace heir
}  // namespace mlir
