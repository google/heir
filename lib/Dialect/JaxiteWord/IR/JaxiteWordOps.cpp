#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"

#include <cstdint>
#include <set>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxiteword {

LogicalResult ProgramInitializationOp::verify() {
  ArrayRef<int64_t> indices = getRotationRequirementIndices();
  ArrayRef<int64_t> offsets = getRotationLevelOffsets();
  if (indices.size() != offsets.size()) {
    return emitOpError()
           << "requires rotationRequirementIndices and rotationLevelOffsets "
              "to have equal lengths";
  }

  std::set<int64_t> totalIndices(getTotalRotationIndices().begin(),
                                 getTotalRotationIndices().end());
  std::set<std::pair<int64_t, int64_t>> requirements;
  for (auto [offset, index] : llvm::zip(offsets, indices)) {
    if (offset < 0) {
      return emitOpError() << "requires nonnegative rotation level offsets";
    }
    if (!totalIndices.contains(index)) {
      return emitOpError() << "requires rotation requirement index " << index
                           << " to appear in totalRotationIndices";
    }
    if (!requirements.insert({offset, index}).second) {
      return emitOpError() << "contains duplicate rotation requirement ("
                           << offset << ", " << index << ")";
    }
  }
  return success();
}

LogicalResult CompiledRegionOp::verify() {
  if (!llvm::hasSingleElement(getBody())) {
    return emitOpError() << "requires exactly one body block";
  }
  Block& block = getBody().front();
  if (block.getNumArguments() != getInputs().size()) {
    return emitOpError() << "requires one block argument per input";
  }
  for (auto [input, argument] : llvm::zip(getInputs(), block.getArguments())) {
    if (input.getType() != argument.getType()) {
      return emitOpError()
             << "requires input and block argument types to match";
    }
  }
  auto yield = dyn_cast<CompiledRegionYieldOp>(block.getTerminator());
  if (!yield) {
    return emitOpError()
           << "requires a jaxiteword.compiled_region_yield terminator";
  }
  if (yield.getValues().getTypes() != getOutputs().getTypes()) {
    return emitOpError() << "requires yielded types to match result types";
  }
  return success();
}

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
