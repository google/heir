#include "lib/Dialect/JaxiteWord/Transforms/JaxiteCkksParameterSelection.h"

#include "lib/Dialect/JaxiteWord/IR/JaxiteWordAttributes.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace jaxiteword {

#define GEN_PASS_DEF_JAXITECKKSPARAMETERSELECTION
#include "lib/Dialect/JaxiteWord/Transforms/Passes.h.inc"

struct JaxiteCkksParameterSelection
    : impl::JaxiteCkksParameterSelectionBase<JaxiteCkksParameterSelection> {
  using JaxiteCkksParameterSelectionBase::JaxiteCkksParameterSelectionBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    SmallVector<int64_t, 2> qTowers = {1, 2};
    SmallVector<int64_t, 1> pTowers = {3};

    auto qTowersAttr = DenseI64ArrayAttr::get(context, qTowers);
    auto pTowersAttr = DenseI64ArrayAttr::get(context, pTowers);

    auto ckksParamsAttr = CkksParametersAttr::get(context, qTowersAttr,
                                                  pTowersAttr, 4, 5, 6, 7, 8);

    module->setAttr("jaxiteword.ckks_params", ckksParamsAttr);
  }
};

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
