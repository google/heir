#include "lib/Dialect/Kernel/Transforms/PrepareLinearTransforms.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

#include "lib/Dialect/Kernel/IR/KernelOps.h"
#include "lib/Dialect/Kernel/IR/KernelTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Target/CompilationTarget/CompilationTarget.h"
#include "mlir/include/mlir/IR/Builders.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"    // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

#define GEN_PASS_DEF_PREPARELINEARTRANSFORMS
#include "lib/Dialect/Kernel/Transforms/Passes.h.inc"

namespace {

// Returns the (possibly tensor-wrapped) LWE ciphertext type of a linear
// transform's input, or null when the input has some other type.
lwe::LWECiphertextType getInputCiphertextType(Type inputType) {
  if (auto ctType = dyn_cast<lwe::LWECiphertextType>(inputType)) return ctType;
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
    return dyn_cast<lwe::LWECiphertextType>(tensorType.getElementType());
  }
  return nullptr;
}

struct PrepareLinearTransforms
    : impl::PrepareLinearTransformsBase<PrepareLinearTransforms> {
  using PrepareLinearTransformsBase::PrepareLinearTransformsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto target = getTargetConfig(module);
    if (failed(target) || !target->has_prepared_linear_transform) return;
    // Only the CKKS lowerings implement prepare/apply, so splitting anything
    // else would leave ops no backend pattern can convert.
    if (!moduleIsCKKS(module)) return;

    module->walk([&](LinearTransformOp op) {
      lwe::LWECiphertextType ctType =
          getInputCiphertextType(op.getInput().getType());
      if (!ctType) return;
      std::optional<int64_t> level = lwe::getLevel(ctType);
      if (!level.has_value()) return;

      // The slot count the diagonals are encoded for is the ciphertext's
      // *encoded* width, which getEncodedSlotCount derives from the ring's
      // capacity and the module's requested count.
      auto plaintextSpace = ctType.getPlaintextSpace();
      int64_t ringCapacity = plaintextSpace.getRing()
                                 .getPolynomialModulus()
                                 .getPolynomial()
                                 .getDegree();
      if (isa<lwe::InverseCanonicalEncodingAttr>(
              plaintextSpace.getEncoding())) {
        ringCapacity /= 2;
      }
      int64_t slots = getEncodedSlotCount(module, ringCapacity);

      OpBuilder builder(op);
      // kernel.linear_transform's bsgs_ratio is a baby-step/giant-step
      // ratio, of which the prepared type records the log2. No attribute
      // means the backend's own default split.
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
      auto preparedType = PreparedLinearTransformType::get(
          module.getContext(), *level, slots, logBsgsRatio);
      auto prepare = PrepareLinearTransformOp::create(
          builder, op.getLoc(), preparedType, op.getDiagonals(),
          op.getDiagonalIndicesAttr(), op.getSourceRowIndicesAttr());
      auto apply = ApplyLinearTransformOp::create(
          builder, op.getLoc(), op.getOutput().getType(), op.getInput(),
          prepare.getPrepared());
      // The transform op may carry analysis attributes (mgmt levels, debug
      // names); they describe the ciphertext computation, so they move to the
      // apply.
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
