#include "lib/Dialect/Secret/Transforms/ImportExecutionResult.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"             // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

// this is in correspondence with insertExternalCall in AddDebugPort.cpp
LogicalResult annotateResult(secret::GenericOp op,
                             const std::vector<std::vector<double>>& data,
                             DataFlowSolver& solver) {
  auto getArrayOfDoubleAttr = [&](const std::vector<double>& row) {
    auto type = Float64Type::get(op.getContext());
    SmallVector<Attribute> elements =
        llvm::to_vector<4>(llvm::map_range(row, [&](double value) -> Attribute {
          return mlir::FloatAttr::get(type, value);
        }));
    return ArrayAttr::get(op.getContext(), elements);
  };

  int lineCount = 0;
  auto tryGetLine = [&]() -> Attribute {
    if (lineCount >= data.size()) {
      return Attribute();
    }
    return getArrayOfDoubleAttr(data[lineCount++]);
  };

  // insert for each argument
  for (auto i = 0; i != op.getBody()->getNumArguments(); ++i) {
    if (!isSecret(op.getBody()->getArgument(i), &solver)) {
      continue;
    }
    auto lineAttr = tryGetLine();
    if (!lineAttr) {
      op.emitError("Not enough data for argument " + std::to_string(i));
      return failure();
    }
    op.setOperandAttr(i, SecretDialect::kArgExecutionResultAttrName, lineAttr);
  }

  // insert after each op
  op.walk([&](Operation* op) {
    if (mlir::isa<secret::GenericOp>(op)) {
      return;
    }

    if (op->getNumResults() == 0) {
      return;
    }

    if (op->getNumResults() > 1) {
      op->emitError("Not supported yet");
      return;
    }
    if (!isSecret(op->getResult(0), &solver)) {
      return;
    }
    // The above lines must come before the attempt to get the line, since
    // reading the line seeks forward in the file pointer and we don't have a
    // way to seek back if we decide to skip it.

    // assume single result for each op!!!
    auto lineAttr = tryGetLine();
    if (!lineAttr) {
      op->emitError("Not enough data for result");
      return;
    }
    op->setAttr(SecretDialect::kArgExecutionResultAttrName, lineAttr);
  });
  return success();
}

#define GEN_PASS_DEF_SECRETIMPORTEXECUTIONRESULT
#include "lib/Dialect/Secret/Transforms/Passes.h.inc"

struct ImportExecutionResult
    : impl::SecretImportExecutionResultBase<ImportExecutionResult> {
  using SecretImportExecutionResultBase::SecretImportExecutionResultBase;

  void runOnOperation() override {
    if (fileName.empty()) {
      getOperation()->emitError("No filename provided");
      signalPassFailure();
      return;
    }

    std::ifstream inputFile(fileName);
    if (!inputFile.is_open()) {
      getOperation()->emitError("Failed to open file: " + fileName);
      signalPassFailure();
      return;
    }

    std::vector<std::vector<double>> data;

    std::string line;
    while (std::getline(inputFile, line)) {
      std::istringstream iss(line);
      std::vector<double> row;
      float number;
      while (iss >> number) {
        row.push_back(number);
      }
      data.push_back(row);
    }

    inputFile.close();

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    solver.load<SecretnessAnalysis>();

    auto result = solver.initializeAndRun(getOperation());
    if (failed(result)) {
      getOperation()->emitOpError() << "Failed to run the analysis.\n";
      signalPassFailure();
      return;
    }

    getOperation()->walk([&](secret::GenericOp genericOp) {
      if (failed(annotateResult(genericOp, data, solver))) {
        genericOp->emitError("Failed to add debug port for genericOp");
        signalPassFailure();
      }
    });
  }
};

}  // namespace secret
}  // namespace heir
}  // namespace mlir
