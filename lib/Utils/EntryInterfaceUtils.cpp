#include "lib/Utils/EntryInterfaceUtils.h"

#include <optional>
#include <utility>

#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

DictionaryAttr getRoleAttr(func::FuncOp function, StringRef name) {
  return function->getAttrOfType<DictionaryAttr>(name);
}

std::optional<StringRef> getRoleEntry(func::FuncOp function, StringRef name) {
  DictionaryAttr attr = getRoleAttr(function, name);
  if (!attr) return std::nullopt;
  auto entry = dyn_cast_or_null<StringAttr>(attr.get(kClientHelperFuncName));
  if (!entry) return std::nullopt;
  return entry.getValue();
}

FailureOr<unsigned> getRoleIndex(func::FuncOp function, StringRef name) {
  DictionaryAttr attr = getRoleAttr(function, name);
  if (!attr) return failure();
  auto index = dyn_cast_or_null<IntegerAttr>(attr.get(kClientHelperIndex));
  if (!index || index.getInt() < 0)
    return function.emitOpError() << name << " is missing a valid index";
  return static_cast<unsigned>(index.getInt());
}

ArrayAttr getLogicalTypes(func::FuncOp function, StringRef name) {
  return function->getAttrOfType<ArrayAttr>(name);
}

func::FuncOp findIndexedHelper(
    ArrayRef<std::pair<unsigned, func::FuncOp>> helpers, unsigned index) {
  for (const auto& [helperIndex, function] : helpers)
    if (helperIndex == index) return function;
  return {};
}

LogicalResult validateIndexedHelpers(
    ArrayRef<std::pair<unsigned, func::FuncOp>> helpers, unsigned count,
    StringRef kind, Operation* diagnostic) {
  std::optional<unsigned> previous;
  for (auto [index, function] : helpers) {
    if (index >= count)
      return function.emitOpError()
             << kind << " index " << index << " is outside the entry signature";
    if (previous == index)
      return diagnostic->emitError()
             << "multiple " << kind << " helpers for index " << index;
    previous = index;
  }
  return success();
}

FailureOr<EntryFunctions> findEntryFunctions(ModuleOp module,
                                             StringRef requestedEntry) {
  // Anchor on the contract when there is one. A program entered below the
  // secret level never ran --add-client-interface and so has no contract; the
  // setup function the backend generated still names its entry.
  SmallVector<func::FuncOp> anchors;
  StringRef anchorRole = kEntryFuncAttrName;
  module.walk([&](func::FuncOp function) {
    if (function->hasAttr(kEntryFuncAttrName)) anchors.push_back(function);
  });
  if (anchors.empty()) {
    anchorRole = kClientSetupFuncAttrName;
    module.walk([&](func::FuncOp function) {
      if (function->hasAttr(kClientSetupFuncAttrName))
        anchors.push_back(function);
    });
  }
  if (anchors.empty())
    return module.emitError(
        "missing a function with heir.entry_func or client.setup_func");

  func::FuncOp anchor;
  if (!requestedEntry.empty()) {
    for (func::FuncOp candidate : anchors) {
      if (getRoleEntry(candidate, anchorRole) == requestedEntry) {
        anchor = candidate;
        break;
      }
    }
    if (!anchor)
      return module.emitError() << "no entry interface for @" << requestedEntry;
  } else {
    if (anchors.size() != 1)
      return module.emitError(
          "multiple entry interfaces require the entry-function option");
    anchor = anchors.front();
  }

  std::optional<StringRef> entry = getRoleEntry(anchor, anchorRole);
  if (!entry)
    return anchor.emitOpError() << anchorRole << " is missing func_name";

  EntryFunctions functions;
  functions.entryName = entry->str();
  if (anchorRole == kEntryFuncAttrName) functions.contract = anchor;
  LogicalResult collectionResult = success();
  module.walk([&](func::FuncOp function) {
    auto matches = [&](StringRef role) {
      return getRoleEntry(function, role) == StringRef(functions.entryName);
    };
    auto collectIndexed =
        [&](StringRef role,
            SmallVector<std::pair<unsigned, func::FuncOp>>& helpers) {
          FailureOr<unsigned> index = getRoleIndex(function, role);
          if (failed(index)) {
            collectionResult = failure();
            return;
          }
          helpers.emplace_back(*index, function);
        };

    if (matches(kClientSetupFuncAttrName)) functions.setup = function;
    if (matches(kClientKeygenFuncAttrName)) functions.keygen = function;
    if (matches(kServerPreprocessingFuncAttrName))
      functions.preprocess = function;
    if (matches(kServerEvaluateFuncAttrName)) functions.evaluate = function;
    for (StringRef role : {kClientEncFuncAttrName, kClientPackFuncAttrName}) {
      if (!matches(role)) continue;
      // An unindexed client.pack_func is an outlined layout helper.
      if (role == kClientPackFuncAttrName &&
          !getRoleAttr(function, role).get(kClientHelperIndex))
        continue;
      collectIndexed(role, functions.inputHelpers);
    }
    if (matches(kClientDecFuncAttrName))
      collectIndexed(kClientDecFuncAttrName, functions.outputHelpers);
    if (matches(kClientEncZeroFuncAttrName))
      collectIndexed(kClientEncZeroFuncAttrName, functions.zeroHelpers);
  });

  if (failed(collectionResult)) return failure();
  auto byIndex = [](const auto& lhs, const auto& rhs) {
    return lhs.first < rhs.first;
  };
  llvm::sort(functions.inputHelpers, byIndex);
  llvm::sort(functions.outputHelpers, byIndex);
  llvm::sort(functions.zeroHelpers, byIndex);

  if (!functions.evaluate)
    return module.emitError()
           << "entry @" << functions.entryName << " has no evaluate function";
  return functions;
}

}  // namespace heir
}  // namespace mlir
