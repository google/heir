#include "lib/Target/OpenFheEmitC/OpenFheEmitCEmitter.h"

#include "mlir/include/mlir/Conversion/ConvertToEmitC/ConvertToEmitCPass.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"         // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"              // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Target/Cpp/CppEmitter.h"  // from @llvm-project

namespace mlir::heir::openfhe {

LogicalResult translateToOpenFheEmitC(Operation* op, llvm::raw_ostream& os) {
  OwningOpRef<Operation*> cloned(op->clone());

  // Sanitize: erase nested pybind modules
  if (auto moduleOp = dyn_cast<ModuleOp>(cloned.get())) {
    SmallVector<ModuleOp> toErase;
    for (auto nestedModule : moduleOp.getOps<ModuleOp>()) {
      if (nestedModule->hasAttr("heir.pybind_module")) {
        toErase.push_back(nestedModule);
      }
    }
    for (auto nestedModule : toErase) {
      nestedModule.erase();
    }
  }

  PassManager pm(cloned.get()->getContext(),
                 cloned.get()->getName().getStringRef());
  pm.addPass(createConvertToEmitC());
  if (failed(pm.run(cloned.get()))) {
    return failure();
  }
  return mlir::emitc::translateToCpp(cloned.get(), os);
}

}  // namespace mlir::heir::openfhe
