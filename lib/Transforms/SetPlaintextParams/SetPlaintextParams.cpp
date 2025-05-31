#include "lib/Transforms/SetPlaintextParams/SetPlaintextParams.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/APInt.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SETPLAINTEXTPARAMS
#include "lib/Transforms/SetPlaintextParams/SetPlaintextParams.h.inc"

struct SetPlaintextParams : impl::SetPlaintextParamsBase<SetPlaintextParams> {
  using SetPlaintextParamsBase::SetPlaintextParamsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    getOperation()->setAttr(
        kPlaintextSchemeAttrName,
        IntegerAttr::get(IntegerType::get(context, 64), logScale));
  }
};

}  // namespace heir
}  // namespace mlir
