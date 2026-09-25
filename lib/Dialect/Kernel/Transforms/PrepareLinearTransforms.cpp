#include "lib/Dialect/Kernel/Transforms/PrepareLinearTransforms.h"

#include <cmath>
#include <cstdint>
#include <optional>

#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/Kernel/IR/KernelTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "mlir/include/mlir/IR/Builders.h"       // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

#define GEN_PASS_DEF_PREPARELINEARTRANSFORMS
#include "lib/Dialect/Kernel/Transforms/Passes.h.inc"

namespace {

struct PrepareLinearTransforms
    : impl::PrepareLinearTransformsBase<PrepareLinearTransforms> {
  using PrepareLinearTransformsBase::PrepareLinearTransformsBase;

  int64_t getNumSlots(ModuleOp module, lwe::LWECiphertextType ctType) {
    auto requestedSlotAttr =
        module->getAttrOfType<IntegerAttr>(kRequestedSlotCountAttrName);
    if (requestedSlotAttr) {
      return requestedSlotAttr.getInt();
    }
    auto plaintextSpace = ctType.getPlaintextSpace();
    int64_t ringCapacity = plaintextSpace.getRing()
                               .getPolynomialModulus()
                               .getPolynomial()
                               .getDegree();
    if (isa<lwe::InverseCanonicalEncodingAttr>(plaintextSpace.getEncoding())) {
      ringCapacity /= 2;
    }
    return ringCapacity;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto target = getTargetConfig(module);
    if (failed(target) || !target->has_kernel_linear_transform) return;

    // Limit to CKKS for now
    if (!moduleIsCKKS(module)) return;

    module->walk([&](LinearTransformOp op) {
      Type eltTy = getElementTypeOrSelf(op.getInput().getType());
      auto ctType = dyn_cast<lwe::LWECiphertextType>(eltTy);
      if (!ctType) return;
      std::optional<int64_t> level = lwe::getLevel(ctType);
      if (!level.has_value()) return;

      OpBuilder builder(op);
      // kernel.linear_transform's bsgs_ratio is a baby-step/giant-step
      // ratio, of which the prepared type records the log2. No attribute
      // means the backend gets to choose the split.
      int64_t logBsgsRatio = 0;
      if (auto ratio = op.getBsgsRatioAttr()) {
        double value = ratio.getValueAsDouble();
        if (value < 1.0) {
          op.emitOpError("bsgs_ratio must be at least 1");
          signalPassFailure();
          return;
        }
        logBsgsRatio = static_cast<int64_t>(std::log2(value));
      }
      int64_t slots = getNumSlots(module, ctType);
      auto preparedType = PreparedLinearTransformType::get(
          module.getContext(), *level, slots, logBsgsRatio);
      auto prepare = PrepareLinearTransformOp::create(
          builder, op.getLoc(), preparedType, op.getDiagonals(),
          op.getDiagonalIndicesAttr(), op.getSourceRowIndicesAttr());
      auto apply = ApplyLinearTransformOp::create(
          builder, op.getLoc(), op.getOutput().getType(), op.getInput(),
          prepare.getPrepared());

      apply->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      op.getOutput().replaceAllUsesWith(apply.getOutput());
      op.erase();
    });
  }
};

}  // namespace

}  // namespace kernel
}  // namespace heir
}  // namespace mlir
