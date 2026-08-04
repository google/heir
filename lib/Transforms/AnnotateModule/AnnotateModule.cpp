#include "lib/Transforms/AnnotateModule/AnnotateModule.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project

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
      BackendName backendEnum = parseBackendName(backend);
      if (backendEnum == BackendName::None) {
        module.emitError() << "Unknown backend: " << backend;
        signalPassFailure();
        return;
      }
      if (!CompilationTargetRegistry::get(backendEnum)) {
        module.emitError() << "Backend not registered: " << backend;
        signalPassFailure();
        return;
      }
      switch (backendEnum) {
        case BackendName::None:
          llvm_unreachable("None should be handled by check above");
        case BackendName::OpenFHE:
          moduleSetOpenfhe(module);
          break;
        case BackendName::Lattigo:
          moduleSetLattigo(module);
          break;
      }
    }
  }
};

}  // namespace heir
}  // namespace mlir
