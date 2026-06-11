#include "lib/Transforms/AnnotatePreprocessing/AnnotatePreprocessing.h"

#include <algorithm>
#include <cstdint>

#include "lib/Analysis/PreprocessingAnalysis/PreprocessingAnalysis.h"
#include "lib/Dialect/HEIRInterfaces.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                    // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ANNOTATEPREPROCESSING
#include "lib/Transforms/AnnotatePreprocessing/AnnotatePreprocessing.h.inc"

struct AnnotatePreprocessing
    : impl::AnnotatePreprocessingBase<AnnotatePreprocessing> {
  using AnnotatePreprocessingBase::AnnotatePreprocessingBase;

  void runOnOperation() override {
    int32_t encodeId = 0;
    getOperation()->walk([&](Operation* op) {
      if (isa<PlaintextEncodeOpInterface>(op)) {
        op->setAttr(
            "encode_id",
            IntegerAttr::get(IntegerType::get(op->getContext(), 32), encodeId));
        encodeId++;
      }
    });

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    SymbolTableCollection symbolTable;
    solver.load<PreprocessingAnalysis>(symbolTable);

    if (failed(solver.initializeAndRun(getOperation()))) {
      getOperation()->emitOpError() << "Failed to run PreprocessingAnalysis.\n";
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](Operation* op) {
      SmallVector<int32_t, 4> encodeIds;
      ValueRange valuesToInspect = op->getNumResults() > 0
                                       ? ValueRange(op->getResults())
                                       : ValueRange(op->getOperands());
      for (Value val : valuesToInspect) {
        auto* lattice = solver.lookupState<PreprocessingLattice>(val);
        if (lattice && lattice->getValue().isInitialized()) {
          for (Operation* encOp : lattice->getValue().getEncodeOps()) {
            if (encOp) {
              if (auto attr = encOp->getAttrOfType<IntegerAttr>("encode_id")) {
                encodeIds.push_back(attr.getInt());
              }
            }
          }
        }
      }

      llvm::sort(encodeIds);
      encodeIds.erase(std::unique(encodeIds.begin(), encodeIds.end()),
                      encodeIds.end());

      if (!encodeIds.empty()) {
        SmallVector<Attribute, 4> attrList;
        attrList.reserve(encodeIds.size());
        for (int32_t id : encodeIds) {
          attrList.push_back(
              IntegerAttr::get(IntegerType::get(op->getContext(), 32), id));
        }
        op->setAttr("downstream_encodes",
                    ArrayAttr::get(op->getContext(), attrList));
      } else {
        op->removeAttr("downstream_encodes");
      }
    });
  }
};

}  // namespace heir
}  // namespace mlir
