#include "lib/Dialect/Rotom/Transforms/MaterializeTensorExtLayout/MaterializeTensorExtLayout.h"

#include <string>

#include "lib/Dialect/Rotom/IR/RotomAttributes.h"
#include "lib/Dialect/Rotom/Utils/RotomTensorExtLayoutLowering.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

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
    getOperation()->walk([](Operation* op) {
      op->removeAttr("rotom.seed");
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        for (unsigned i = 0; i < func.getNumArguments(); ++i) {
          func.removeArgAttr(i, "rotom.seed");
        }
        for (unsigned i = 0; i < func.getNumResults(); ++i) {
          func.removeResultAttr(i, "rotom.seed");
        }
      }
    });
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

    // A layouted PUBLIC value produced by cleartext compute -- e.g. a bias
    // that flows through host-side arithmetic before entering the secret
    // region -- is an encode-time packing boundary. Producers with no
    // layouts stay cleartext, so the packing must be stated explicitly as a
    // tensor_ext.assign_layout for the ciphertext-semantics conversion (which
    // otherwise has no way to convert the producer chain).
    StringRef tensorExtLayoutAttrName =
        tensor_ext::TensorExtDialect::kLayoutAttrName;
    module.walk([&](Operation* op) {
      if (isa<tensor_ext::AssignLayoutOp>(op)) return;
      if (op->getParentOfType<secret::GenericOp>()) return;
      if (op->getNumResults() != 1) return;
      Value value = op->getResult(0);
      if (!isa<RankedTensorType>(value.getType())) return;
      FailureOr<Attribute> layoutAttr =
          findAttributeAssociatedWith(value, tensorExtLayoutAttrName);
      if (failed(layoutAttr)) return;
      auto layout = dyn_cast<tensor_ext::LayoutAttr>(*layoutAttr);
      if (!layout) return;
      // The boundary is where no operand carries a layout: a producer with a
      // layouted operand converts as a normal layout-carrying op. A
      // zero-operand producer (arith.constant) is always a boundary.
      for (Value operand : op->getOperands()) {
        if (!isa<RankedTensorType>(operand.getType())) continue;
        if (succeeded(findAttributeAssociatedWith(operand,
                                                  tensorExtLayoutAttrName))) {
          return;
        }
      }

      OpBuilder builder(op->getContext());
      builder.setInsertionPointAfter(op);
      auto assign = tensor_ext::AssignLayoutOp::create(builder, op->getLoc(),
                                                       value, layout);
      setAttributeAssociatedWith(assign.getOutput(), tensorExtLayoutAttrName,
                                 layout);
      value.replaceAllUsesExcept(assign.getOutput(), assign);
      removeAttributeAssociatedWith(value, tensorExtLayoutAttrName);
    });

    module.walk([&](func::FuncOp func) {
      for (int64_t i = 0; i < func.getNumResults(); ++i) {
        auto layout = dyn_cast_or_null<LayoutAttr>(
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

    // A secret.generic operand has its layout recorded on the operand
    // annotation. The downstream type converter propagates the now-materialized
    // operand layout onto the function argument.
    StringRef tensorExtLayoutName =
        tensor_ext::TensorExtDialect::kLayoutAttrName;
    module.walk([&](secret::GenericOp gen) {
      for (OpOperand& operand : gen->getOpOperands()) {
        auto funcArg = dyn_cast<BlockArgument>(operand.get());
        if (!funcArg ||
            !isa<FunctionOpInterface>(funcArg.getOwner()->getParentOp())) {
          continue;
        }
        BlockArgument blockArg =
            gen.getRegion().getArgument(operand.getOperandNumber());
        FailureOr<Attribute> operandLayout =
            findAttributeAssociatedWith(blockArg, tensorExtLayoutName);
        if (failed(operandLayout)) continue;

        FailureOr<Attribute> existing =
            findAttributeAssociatedWith(funcArg, tensorExtLayoutName);
        if (succeeded(existing)) {
          if (*existing != *operandLayout) {
            gen.emitError()
                << "function argument " << funcArg.getArgNumber()
                << " feeds secret.generic operands with conflicting "
                   "materialized layouts";
            result = failure();
          }
          continue;
        }
        setAttributeAssociatedWith(funcArg, tensorExtLayoutName,
                                   *operandLayout);
      }
    });

    if (failed(result)) signalPassFailure();
  }
};

}  // namespace mlir::heir::rotom
