#include "lib/Utils/Utils.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {

Operation* walkAndDetect(Operation* op, OpPredicate predicate) {
  Operation* foundOp = nullptr;
  op->walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (predicate(op)) {
      foundOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return foundOp;
}

LogicalResult validateValues(Operation* op, IsValidValueFn isValidValue) {
  bool argRes = llvm::all_of(op->getOperands(), [&](Value value) {
    return succeeded(isValidValue(value));
  });
  return !argRes ? failure() : success();
}

LogicalResult validateTypes(Operation* op, IsValidTypeFn isValidType) {
  bool argRes = llvm::all_of(op->getOperandTypes(), [&](Type type) {
    return succeeded(isValidType(type));
  });
  bool resultRes = llvm::all_of(op->getResultTypes(), [&](Type type) {
    return succeeded(isValidType(type));
  });
  return (!argRes || !resultRes) ? failure() : success();
}

LogicalResult walkAndValidateValues(Operation* op, IsValidValueFn isValidValue,
                                    std::optional<std::string> err) {
  LogicalResult res = success();
  op->walk([&](Operation* op) {
    res = validateValues(op, isValidValue);
    if (failed(res) && err.has_value()) op->emitError() << err.value();
    return failed(res) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return res;
}

void walkValues(Operation* op, ValueProcessor valueProcessor) {
  op->walk([&](Operation* op) {
    // Region-holding ops may not have operands or results, but their region's
    // arguments are BlockArguments that may not be processed by any other
    // means, e.g., an unused function argument, cf.
    // https://github.com/google/heir/issues/1406
    if (op->getNumRegions() > 0) {
      for (auto& region : op->getRegions()) {
        for (auto& block : region.getBlocks()) {
          for (auto arg : block.getArguments()) {
            valueProcessor(arg);
          }
        }
      }
    }

    if (op->getNumResults() > 0) {
      for (auto result : op->getResults()) {
        valueProcessor(result);
      }
    }

    if (op->getNumOperands() > 0) {
      for (auto operand : op->getOperands()) {
        valueProcessor(operand);
      }
    }
  });
}

bool containsArgumentOfType(Operation* op, TypePredicate predicate) {
  // special treatment for func declaration
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    return llvm::any_of(funcOp.getArgumentTypes(),
                        [&](Type type) { return predicate(type); });
  }
  return llvm::any_of(op->getRegions(), [&](Region& region) {
    return llvm::any_of(region.getBlocks(), [&](Block& block) {
      return llvm::any_of(block.getArguments(), [&](BlockArgument arg) {
        return predicate(arg.getType());
      });
    });
  });
}

void iterateIndices(ArrayRef<int64_t> shape, const IndexTupleConsumer& process,
                    ArrayRef<int64_t> fixedIndices,
                    ArrayRef<int64_t> fixedIndexValues) {
  if (shape.empty()) return;
  assert(fixedIndices.size() == fixedIndexValues.size() &&
         "size mismatch on fixed indices");
  for (size_t i = 0; i < fixedIndices.size(); ++i) {
    assert(fixedIndices[i] >= 0 ||
           fixedIndices[i] < static_cast<int64_t>(shape.size()) &&
               "Fixed index is out of range");
    assert(fixedIndexValues[i] >= 0 ||
           fixedIndexValues[i] < shape[fixedIndices[i]] &&
               "Fixed index value is out of range");
  }

  std::map<int64_t, int64_t> fixedIndexToValue;
  for (size_t i = 0; i < fixedIndices.size(); ++i) {
    fixedIndexToValue[fixedIndices[i]] = fixedIndexValues[i];
  }

  auto isFixed = [&fixedIndexToValue](int64_t idx) -> bool {
    return fixedIndexToValue.count(idx) > 0;
  };

  std::vector<int64_t> indices(shape.size(), 0);
  // Pre-populate with fixed indices
  for (size_t i = 0; i < fixedIndices.size(); ++i) {
    indices[fixedIndices[i]] = fixedIndexValues[i];
  }

  bool done = false;

  while (!done) {
    // Process current indices
    process(indices);

    int dim = shape.size() - 1;
    while (dim >= 0) {
      if (!isFixed(dim)) {
        indices[dim]++;
        if (indices[dim] < shape[dim]) {
          // No carry needed
          break;
        }
        // Reset this digit and move to the next
        indices[dim] = 0;
      }
      dim--;
    }

    // If we've decremented past the first dimension, we're done
    if (dim < 0) {
      done = true;
    }
  }
}

std::string doubleToString2Prec(double value) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << value;
  return stream.str();
}

}  // namespace heir
}  // namespace mlir
