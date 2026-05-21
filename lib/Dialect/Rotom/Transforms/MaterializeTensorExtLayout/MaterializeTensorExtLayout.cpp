#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h"

#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
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

    auto lowerLayout = [&](Location loc,
                           LayoutAttr layout) -> FailureOr<Attribute> {
      FailureOr<std::string> isl =
          RotomTensorExtLayoutLowering::lowerToTensorExtIsl(layout);
      if (failed(isl)) {
        emitError(loc, "unsupported rotom.layout for materialization");
        return failure();
      }
      return tensor_ext::LayoutAttr::get(module.getContext(), *isl);
    };

    walkValues(module, [&](Value value) {
      FailureOr<Attribute> rotomAttr =
          findAttributeAssociatedWith(value, kRotomLayoutAttrName);
      if (failed(rotomAttr)) return;

      auto layout = dyn_cast<LayoutAttr>(*rotomAttr);
      if (!layout) return;

      FailureOr<Attribute> tensorExtLayout =
          lowerLayout(value.getLoc(), layout);
      if (failed(tensorExtLayout)) {
        result = failure();
        return;
      }

      setAttributeAssociatedWith(value,
                                 tensor_ext::TensorExtDialect::kLayoutAttrName,
                                 *tensorExtLayout);
      removeAttributeAssociatedWith(value, kRotomLayoutAttrName);
    });

    module.walk([&](func::FuncOp func) {
      for (int64_t i = 0; i < func.getNumResults(); ++i) {
        auto layout =
            dyn_cast_or_null<LayoutAttr>(
                func.getResultAttr(i, kRotomLayoutAttrName));
        if (!layout) continue;

        FailureOr<Attribute> tensorExtLayout =
            lowerLayout(func.getLoc(), layout);
        if (failed(tensorExtLayout)) {
          result = failure();
          return;
        }

        func.setResultAttr(i, tensor_ext::TensorExtDialect::kLayoutAttrName,
                           *tensorExtLayout);
        func.removeResultAttr(i, kRotomLayoutAttrName);
      }
    });

    if (failed(result)) signalPassFailure();
  }
};

}  // namespace mlir::heir::rotom
