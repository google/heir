#ifndef LIB_TARGET_COMPILATIONTARGET_COMPILATIONTARGET_H_
#define LIB_TARGET_COMPILATIONTARGET_COMPILATIONTARGET_H_

// IWYU pragma: begin_keep
#include <cstdint>
#include <string>
// IWYU pragma: end_keep

#include "llvm/include/llvm/ADT/StringMap.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// Include after earlier includes
#include "lib/Target/CompilationTarget/CompilationTargetStruct.h.inc"

namespace mlir {
namespace heir {

class CompilationTargetRegistry {
 public:
  static const CompilationTarget* get(llvm::StringRef name);
  static void registerTarget(const CompilationTarget& target);

 private:
  CompilationTargetRegistry();
  static CompilationTargetRegistry& getInstance();

  llvm::StringMap<CompilationTarget> targets;
};

// Return the CompilationTarget corresponding to the backend registered as a
// module attribute (e.g., backend.lattigo) along with any overrides applied via
// the backend.config_override dictionary attr. Returns failure if the backend
// resolved was not registered, or if any overrides fail to be applied (for
// example, due to an incorrectly spelled key).
FailureOr<CompilationTarget> getTargetConfig(ModuleOp module);

// Validate that the key-value pair is supported and valid for the target config
// of the module.
LogicalResult validateCompilationTargetOverride(ModuleOp module, StringRef key,
                                                Attribute value);

// Persist the override to the module. If the module already has an override
// dictionary, it merges the new override into it.
void persistCompilationTargetOverride(ModuleOp module, StringRef key,
                                      Attribute value);

// Convert a string value to an Attribute of the appropriate type for the key.
// Returns failure if the key is not supported or the value cannot be parsed.
FailureOr<Attribute> parseCompilationTargetOverrideValue(MLIRContext* ctx,
                                                         StringRef key,
                                                         StringRef value);

}  // namespace heir
}  // namespace mlir
#endif  // LIB_TARGET_COMPILATIONTARGET_COMPILATIONTARGET_H_
