#include <string>

// IWYU pragma: begin_keep
#include "lib/Transforms/AnnotateModule/AnnotateModule.h"
// IWYU pragma: end_keep

#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "llvm/include/llvm/ADT/StringRef.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_APPLYCONFIGOVERRIDE
#include "lib/Transforms/AnnotateModule/AnnotateModule.h.inc"

struct ApplyConfigOverride
    : impl::ApplyConfigOverrideBase<ApplyConfigOverride> {
  using ApplyConfigOverrideBase::ApplyConfigOverrideBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    MLIRContext* ctx = &getContext();

    for (const std::string& kv : config) {
      StringRef kvRef(kv);
      auto split = kvRef.split('=');
      if (split.second.empty() && !kvRef.contains('=')) {
        module.emitError() << "Invalid config format, expected key=value: "
                           << kv;
        signalPassFailure();
        return;
      }
      StringRef key = split.first.trim();
      StringRef value = split.second.trim();

      FailureOr<Attribute> attr =
          parseCompilationTargetOverrideValue(ctx, key, value);
      if (failed(attr)) {
        module.emitError() << "Failed to parse value for key " << key << ": "
                           << value;
        signalPassFailure();
        return;
      }

      if (failed(validateCompilationTargetOverride(module, key, *attr))) {
        module.emitError() << "Invalid override for key " << key << ": "
                           << value;
        signalPassFailure();
        return;
      }

      persistCompilationTargetOverride(module, key, *attr);
    }
  }
};

}  // namespace heir
}  // namespace mlir
