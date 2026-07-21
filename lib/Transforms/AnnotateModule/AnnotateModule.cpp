#include "lib/Transforms/AnnotateModule/AnnotateModule.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"   // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ANNOTATEMODULE
#include "lib/Transforms/AnnotateModule/AnnotateModule.h.inc"

struct AnnotateModule : impl::AnnotateModuleBase<AnnotateModule> {
  using AnnotateModuleBase::AnnotateModuleBase;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());

    if (scheme == "bgv") {
      moduleSetBGV(module);
    } else if (scheme == "bfv") {
      moduleSetBFV(module);
    } else if (scheme == "ckks") {
      moduleSetCKKS(module);
    } else if (scheme == "cggi") {
      moduleSetCGGI(module);
    }

    if (!backend.empty()) {
      if (!CompilationTargetRegistry::get(backend)) {
        module.emitError() << "Unknown backend: " << backend;
        signalPassFailure();
        return;
      }

      if (backend == "openfhe") {
        moduleSetOpenfhe(module);
      } else if (backend == "lattigo") {
        moduleSetLattigo(module);
      }
    }
  }
};

}  // namespace heir
}  // namespace mlir
