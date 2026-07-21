#include "lib/Target/CompilationTarget/CompilationTarget.h"

#include <cassert>
#include <string>

#include "lib/Target/CompilationTarget/CompilationTargetOverrides.cpp.inc"
#include "llvm/include/llvm/ADT/StringMap.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

CompilationTargetRegistry::CompilationTargetRegistry() {}

void CompilationTargetRegistry::registerTarget(
    const CompilationTarget& target) {
  auto& instance = getInstance();
  auto [it, inserted] =
      instance.targets.try_emplace(target.backendName, target);
  assert(inserted && "Target already registered");
  (void)inserted;
}

CompilationTargetRegistry& CompilationTargetRegistry::getInstance() {
  static CompilationTargetRegistry instance;
  return instance;
}

const CompilationTarget* CompilationTargetRegistry::get(llvm::StringRef name) {
  auto& instance = getInstance();
  auto it = instance.targets.find(name);
  if (it == instance.targets.end()) {
    return nullptr;
  }
  return &it->second;
}

FailureOr<std::string> findBackend(ModuleOp module) {
  for (NamedAttribute attr : module->getAttrs()) {
    // This skips over backend.config_override, in particular
    if (!isa<UnitAttr>(attr.getValue())) continue;

    llvm::StringRef name = attr.getName().getValue();
    if (name.consume_front("backend.")) {
      return std::string(name);
    }
  }
  return failure();
}

FailureOr<CompilationTarget> getTargetConfig(ModuleOp module) {
  FailureOr<std::string> backend = findBackend(module);
  if (failed(backend)) return failure();

  const CompilationTarget* registered =
      CompilationTargetRegistry::get(*backend);
  if (!registered)
    return module.emitError()
           << "Could not find registered backend with name " << *backend;

  // Eager copy to allow for overrides
  CompilationTarget resolved = *registered;
  if (auto overridesAttr =
          module->getAttrOfType<DictionaryAttr>("backend.config_override")) {
    for (auto namedAttr : overridesAttr) {
      auto key = namedAttr.getName().strref();
      auto value = namedAttr.getValue();
      auto result =
          applyCompilationTargetOverride(resolved, key, value, module.getLoc());
      if (failed(result)) {
        return failure();
      }
    }
  }

  return FailureOr<CompilationTarget>(resolved);
}

LogicalResult validateCompilationTargetOverride(ModuleOp module, StringRef key,
                                                Attribute value) {
  FailureOr<CompilationTarget> target = getTargetConfig(module);
  if (failed(target)) return failure();
  CompilationTarget targetCopy = *target;
  return applyCompilationTargetOverride(targetCopy, key, value,
                                        module.getLoc());
}

void persistCompilationTargetOverride(ModuleOp module, StringRef key,
                                      Attribute value) {
  MLIRContext* ctx = module.getContext();
  DictionaryAttr overrides =
      module->getAttrOfType<DictionaryAttr>("backend.config_override");
  SmallVector<NamedAttribute, 4> newOverrides;
  bool replaced = false;
  if (overrides) {
    for (auto attr : overrides) {
      if (attr.getName() == key) {
        newOverrides.push_back(
            NamedAttribute(StringAttr::get(ctx, key), value));
        replaced = true;
      } else {
        newOverrides.push_back(attr);
      }
    }
  }
  if (!replaced) {
    newOverrides.push_back(NamedAttribute(StringAttr::get(ctx, key), value));
  }
  module->setAttr("backend.config_override",
                  DictionaryAttr::get(ctx, newOverrides));
}

}  // namespace heir

}  // namespace mlir
