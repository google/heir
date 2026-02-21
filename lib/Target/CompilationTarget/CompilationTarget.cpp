#include "lib/Target/CompilationTarget/CompilationTarget.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/StringMap.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir {
namespace heir {

#include "lib/Target/CompilationTarget/CompilationTarget.cpp.inc"

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

const CompilationTarget* getTargetConfig(ModuleOp module) {
  for (auto attr : module->getAttrs()) {
    llvm::StringRef name = attr.getName().strref();
    if (name.consume_front("backend.")) {
      return CompilationTargetRegistry::get(name);
    }
  }
  return nullptr;
}

}  // namespace heir
}  // namespace mlir
