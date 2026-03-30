#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h"

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir::heir::rotom {

namespace {
constexpr llvm::StringLiteral kRotomLayoutAttrName = "rotom.layout";
}

#define GEN_PASS_DEF_MATERIALIZETENSOREXTLAYOUT
#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h.inc"

struct MaterializeTensorExtLayout
    : public impl::MaterializeTensorExtLayoutBase<MaterializeTensorExtLayout> {
  using MaterializeTensorExtLayoutBase::MaterializeTensorExtLayoutBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    LogicalResult result = success();

    walkValues(module, [&](Value value) {
      FailureOr<Attribute> rotomAttr =
          findAttributeAssociatedWith(value, kRotomLayoutAttrName);
      if (failed(rotomAttr)) return;

      auto layout = dyn_cast<LayoutAttr>(*rotomAttr);
      if (!layout) return;

      FailureOr<std::string> isl =
          RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
      if (failed(isl)) {
        emitError(value.getLoc(),
                  "unsupported rotom.layout for materialization");
        result = failure();
        return;
      }

      auto tensorExtLayout =
          tensor_ext::LayoutAttr::get(module.getContext(), *isl);
      setAttributeAssociatedWith(value,
                                 tensor_ext::TensorExtDialect::kLayoutAttrName,
                                 tensorExtLayout);
      removeAttributeAssociatedWith(value, kRotomLayoutAttrName);
    });

    if (failed(result)) signalPassFailure();
  }
};

}  // namespace mlir::heir::rotom
