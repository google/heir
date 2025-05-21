#include "lib/Dialect/Mgmt/Transforms/AnnotateMgmt.h"

#include "lib/Analysis/DimensionAnalysis/DimensionAnalysis.h"
#include "lib/Analysis/LevelAnalysis/LevelAnalysis.h"
#include "lib/Analysis/ScaleAnalysis/ScaleAnalysis.h"
#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/Mgmt/Transforms/Utils.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Transforms/PropagateAnnotation/PropagateAnnotation.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace mgmt {

#define GEN_PASS_DEF_ANNOTATEMGMT
#include "lib/Dialect/Mgmt/Transforms/Passes.h.inc"

void mergeMgmtAttrs(Operation *top) {
  auto mergeIntoMgmtAttr = [&](Attribute levelAttr, Attribute dimensionAttr,
                               Attribute scaleAttr) {
    auto level = cast<IntegerAttr>(levelAttr).getInt();
    auto dimension = cast<IntegerAttr>(dimensionAttr).getInt();
    if (!scaleAttr) {
      return MgmtAttr::get(top->getContext(), level, dimension);
    }
    auto scale = cast<IntegerAttr>(scaleAttr).getInt();
    return MgmtAttr::get(top->getContext(), level, dimension, scale);
  };

  walkValues(top, [&](Value value) {
    FailureOr<Attribute> levelAttr =
        findAttributeAssociatedWith(value, kArgLevelAttrName);
    FailureOr<Attribute> dimensionAttr =
        findAttributeAssociatedWith(value, kArgDimensionAttrName);
    if (failed(levelAttr) || failed(dimensionAttr)) {
      return;
    }
    Attribute scaleAttr =
        findAttributeAssociatedWith(value, kArgScaleAttrName).value_or(nullptr);
    Attribute mgmtAttr =
        mergeIntoMgmtAttr(levelAttr.value(), dimensionAttr.value(), scaleAttr);
    setAttributeAssociatedWith(value, MgmtDialect::kArgMgmtAttrName, mgmtAttr);
  });
}

struct AnnotateMgmt : impl::AnnotateMgmtBase<AnnotateMgmt> {
  using AnnotateMgmtBase::AnnotateMgmtBase;

  void runOnOperation() override {
    DataFlowSolver solver;
    SymbolTableCollection symbolTable;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<SecretnessAnalysis>();
    solver.load<LevelAnalysis>();
    solver.load<LevelAnalysisBackward>(symbolTable);
    solver.load<DimensionAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    // This pass works by first running independent analysis passes that add
    // separate annotations (mgmt.level, mgmt.dimension, mgmt.scale) to the IR,
    // and then a final step combines them into a single `mgmt.mgmt` attribute
    // for later passes to use.
    //
    // annotateScale is not included here because the details of scale
    // management are specific enough to warrant their own passes. So each
    // annotate-scale pass must itself run the ScaleAnalysis, annotateScale,
    // and then call this pass in post-processing.
    annotateLevel(getOperation(), &solver, baseLevel);
    annotateDimension(getOperation(), &solver);

    mergeMgmtAttrs(getOperation());
    clearAttrs(getOperation(), kArgLevelAttrName);
    clearAttrs(getOperation(), kArgDimensionAttrName);
    clearAttrs(getOperation(), kArgScaleAttrName);

    // Dataflow analyses don't assign anything to function results because they
    // don't have a corresponding Value. So we have to manually copy it from
    // the func terminator.
    copyReturnOperandAttrsToFuncResultAttrs(getOperation(),
                                            MgmtDialect::kArgMgmtAttrName);

    if (failed(copyAttrToClientHelpers(getOperation(),
                                       MgmtDialect::kArgMgmtAttrName))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace mgmt
}  // namespace heir
}  // namespace mlir
