#include "lib/Utils/OpAsmInterfaceHelper.h"

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"

namespace mlir {
namespace heir {

void suggestNameForValue(Value value, ::mlir::OpAsmSetValueNameFn setNameFn) {
  auto suggestFunctions = {
      lwe::lweSuggestNameForType,
      openfhe::openfheSuggestNameForType,
  };
  // only the first suggestion is used
  // if no suggestion then do nothing
  for (auto suggest : suggestFunctions) {
    auto suggested = suggest(value.getType());
    if (!suggested.empty()) {
      setNameFn(value, suggested);
      return;
    }
  }
}

void getAsmBlockArgumentNames(Operation* op, Region& region,
                              ::mlir::OpAsmSetValueNameFn setNameFn) {
  for (auto& block : region) {
    for (auto arg : block.getArguments()) {
      suggestNameForValue(arg, setNameFn);
    }
  }
}

void getAsmResultNames(Operation* op, ::mlir::OpAsmSetValueNameFn setNameFn) {
  for (auto result : op->getResults()) {
    suggestNameForValue(result, setNameFn);
  }
}

}  // namespace heir
}  // namespace mlir
