#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h"

#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
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

    // A secret.secret function argument used as a secret.generic operand has its
    // layout recorded on the operand annotation (setAttributeAssociatedWith
    // routes a block-arg attribute to the enclosing op's operand), not on the
    // function argument. The downstream type converter reads the FUNCTION
    // argument attribute, so propagate the now-materialized operand layout onto
    // the argument; otherwise the argument type is left unconverted and the
    // secret.generic operand/block-argument types mismatch.
    StringRef tensorExtLayoutName =
        tensor_ext::TensorExtDialect::kLayoutAttrName;
    module.walk([&](secret::GenericOp gen) {
      for (OpOperand& operand : gen->getOpOperands()) {
        auto funcArg = dyn_cast<BlockArgument>(operand.get());
        if (!funcArg) continue;
        auto func = dyn_cast<FunctionOpInterface>(
            funcArg.getOwner()->getParentOp());
        if (!func) continue;
        unsigned argNo = funcArg.getArgNumber();
        BlockArgument blockArg =
            gen.getRegion().getArgument(operand.getOperandNumber());
        FailureOr<Attribute> operandLayout =
            findAttributeAssociatedWith(blockArg, tensorExtLayoutName);
        if (failed(operandLayout)) continue;
        // The type converter rewrites the function argument exactly once from this
        // attribute. If a second secret.generic operand backed by the same
        // argument needs a different layout, the argument cannot satisfy both;
        // fail loudly instead of silently keeping the first (which would leave the
        // other operand's block-argument type mismatched).
        if (Attribute existing = func.getArgAttr(argNo, tensorExtLayoutName)) {
          if (existing != *operandLayout) {
            gen.emitError() << "function argument " << argNo
                            << " feeds secret.generic operands with conflicting "
                               "materialized layouts";
            result = failure();
          }
          continue;
        }
        func.setArgAttr(argNo, tensorExtLayoutName, *operandLayout);
      }
    });

    if (failed(result)) signalPassFailure();
  }
};

}  // namespace mlir::heir::rotom
