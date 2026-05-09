#ifndef LIB_TARGET_COMPILATIONTARGET_COMPILATIONTARGET_H_
#define LIB_TARGET_COMPILATIONTARGET_COMPILATIONTARGET_H_

#include <string>

#include "llvm/include/llvm/ADT/StringMap.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir {
namespace heir {

struct CompilationTarget {
  std::string backendName;
  int bootstrapLevelsConsumed;
};

class CompilationTargetRegistry {
 public:
  static const CompilationTarget* get(llvm::StringRef name);

 private:
  CompilationTargetRegistry();
  static CompilationTargetRegistry& getInstance();

  llvm::StringMap<CompilationTarget> targets;
};

const CompilationTarget* getTargetConfig(ModuleOp module);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_COMPILATIONTARGET_COMPILATIONTARGET_H_
