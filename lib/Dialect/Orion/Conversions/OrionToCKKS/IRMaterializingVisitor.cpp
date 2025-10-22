#include "lib/Dialect/Orion/Conversions/OrionToCKKS/IRMaterializingVisitor.h"

#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/Support/Debug.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

#define DEBUG_TYPE "orion-to-ckks"

namespace mlir {
namespace heir {
namespace orion {

using kernel::AddNode;
using kernel::ConstantScalarNode;
using kernel::ConstantTensorNode;
using kernel::ExtractNode;
using kernel::LeafNode;
using kernel::LeftRotateNode;
using kernel::MultiplyNode;
using kernel::SSAValue;
using kernel::SubtractNode;

Value IRMaterializingVisitor::encodeCleartextOperand(
    lwe::LWECiphertextType ctTy, Value cleartext, bool useDefaultScale) {
  MLIRContext* ctx = builder.getContext();
  int64_t newLogScale = useDefaultScale
                            ? logDefaultScale
                            : lwe::getScalingFactorFromEncodingAttr(
                                  ctTy.getPlaintextSpace().getEncoding());
  if (auto encodeOp = cleartext.getDefiningOp<lwe::RLWEEncodeOp>()) {
    // The cleartext is already encoded, so we need to set its scaling factor
    // to match the ciphertext.
    lwe::LWEPlaintextType oldPtType =
        cast<lwe::LWEPlaintextType>(encodeOp.getResult().getType());
    int64_t oldLogScale = lwe::getScalingFactorFromEncodingAttr(
        oldPtType.getPlaintextSpace().getEncoding());
    lwe::LWEPlaintextType newPtType = getCorrespondingPlaintextType(ctTy);
    encodeOp.setEncodingAttr(newPtType.getPlaintextSpace().getEncoding());
    LLVM_DEBUG(llvm::dbgs() << "Cleartext is already encoded\n");
    if (oldLogScale == newLogScale) {
      LLVM_DEBUG(llvm::dbgs()
                 << "scaling factor already matches: " << newLogScale << "\n");
      return encodeOp.getResult();
    }
    LLVM_DEBUG(llvm::dbgs() << "adjusting scaling factor from "
                            << lwe::getScalingFactorFromEncodingAttr(
                                   oldPtType.getPlaintextSpace().getEncoding())
                            << " to " << newLogScale << "\n");
    encodeOp.getResult().setType(newPtType);
    LLVM_DEBUG(llvm::dbgs() << "New encode op:" << encodeOp << "\n");
    return encodeOp.getResult();
  }

  LLVM_DEBUG(
      llvm::dbgs() << "Cleartext is not encoded; adding encoding for scale "
                   << newLogScale << "\n");
  auto encoding = lwe::InverseCanonicalEncodingAttr::get(ctx, newLogScale);
  auto ring = ctTy.getPlaintextSpace().getRing();

  lwe::LWEPlaintextType ptTy = lwe::LWEPlaintextType::get(
      ctx, ctTy.getApplicationData(),
      lwe::PlaintextSpaceAttr::get(ctx, ring, encoding));
  auto encodeOp =
      lwe::RLWEEncodeOp::create(builder, ptTy, cleartext, encoding, ring)
          .getResult();
  return encodeOp;
}

Value IRMaterializingVisitor::relinAndRescale(Value value, bool relinearize,
                                              bool rescale) {
  Value result = value;
  lwe::LWECiphertextType ctTy = cast<lwe::LWECiphertextType>(value.getType());
  if (relinearize) {
    LLVM_DEBUG(llvm::dbgs() << "Relinearizing ciphertext\n");
    auto inputDimension = cast<lwe::LWECiphertextType>(value.getType())
                              .getCiphertextSpace()
                              .getSize();
    SmallVector<int32_t> fromBasis;
    for (int i = 0; i < inputDimension; ++i) {
      fromBasis.push_back(i);
    }
    SmallVector<int32_t> toBasis = {0, 1};
    auto relinOp = ckks::RelinearizeOp::create(
        builder, result, builder.getDenseI32ArrayAttr(fromBasis),
        builder.getDenseI32ArrayAttr(toBasis));
    result = relinOp.getResult();
    ctTy = cast<lwe::LWECiphertextType>(result.getType());
  }
  if (rescale) {
    LLVM_DEBUG(llvm::dbgs() << "Rescaling ciphertext from current scale: "
                            << lwe::getScalingFactorFromEncodingAttr(
                                   ctTy.getPlaintextSpace().getEncoding())
                            << " \n");
    FailureOr<lwe::LWECiphertextType> ctTypeResult = applyModReduce(ctTy);
    if (failed(ctTypeResult)) {
      emitError(result.getLoc())
          << "Cannot rescale ciphertext type, inserting extra bootstrap op";
      // sub 1 because the max level is the last index in the chain.
      int64_t maxLevel = ctTy.getModulusChain().getElements().size() - 1;

      // Now we cheat a little bit: normally bootstrap itself would consume
      // some levels, which depends on the chosen backend. In our case, we're
      // lowering to library backends that handle this opaquely.
      //
      // TODO(#1207): fix if this pass still matters when lowering to
      // polynomial.
      FailureOr<lwe::LWECiphertextType> outputTypeResult =
          cloneAtLevel(ctTy, maxLevel);
      if (failed(outputTypeResult)) {
        emitError(result.getLoc()) << "Failed to insert bootstrap";
        return Value();
      }
      result =
          ckks::BootstrapOp::create(builder, outputTypeResult.value(), result);
    }
    auto ctType = ctTypeResult.value();
    result = ckks::RescaleOp::create(builder, ctType, result,
                                     ctType.getCiphertextSpace().getRing());
  }
  return result;
}

Value IRMaterializingVisitor::operator()(const LeafNode<SSAValue>& node) {
  return node.value.getValue();
}

Value IRMaterializingVisitor::operator()(const ConstantScalarNode& node) {
  // Create a constant and encode a plaintext with the splatted value in the
  // node.
  LLVM_DEBUG(llvm::dbgs() << "Visiting constant scalar " << node.value << "\n");
  MLIRContext* ctx = builder.getContext();
  int64_t numSlots = plaintextType.getPlaintextSpace()
                         .getRing()
                         .getPolynomialModulus()
                         .getPolynomial()
                         .getDegree() /
                     2;
  RankedTensorType cleartextType =
      RankedTensorType::get({numSlots}, Float32Type::get(ctx));
  auto constantOp = arith::ConstantOp::create(
      builder, getScalarOrDenseAttr(cleartextType, APFloat(node.value)));
  auto encodeOp =
      lwe::RLWEEncodeOp::create(builder, plaintextType, constantOp.getResult(),
                                plaintextType.getPlaintextSpace().getEncoding(),
                                plaintextType.getPlaintextSpace().getRing());
  return encodeOp.getResult();
}

Value IRMaterializingVisitor::operator()(const ConstantTensorNode& node) {
  llvm_unreachable("not supported");
  return Value();
}

Value IRMaterializingVisitor::operator()(const AddNode<SSAValue>& node) {
  return nonMulBinop<AddNode<SSAValue>, ckks::AddOp, ckks::AddPlainOp,
                     arith::AddFOp>(node);
}

Value IRMaterializingVisitor::operator()(const SubtractNode<SSAValue>& node) {
  return nonMulBinop<SubtractNode<SSAValue>, ckks::SubOp, ckks::SubPlainOp,
                     arith::SubFOp>(node);
}

Value IRMaterializingVisitor::operator()(const MultiplyNode<SSAValue>& node) {
  Value lhs = this->process(node.left);
  Value rhs = this->process(node.right);

  // Just dyn_cast to all possibilities, and keep the nesting structure flat
  // to avoid awkward contortions of doing type switches on both lhs and rhs.
  lwe::LWECiphertextType lhsCiphertextType =
      dyn_cast<lwe::LWECiphertextType>(lhs.getType());
  lwe::LWECiphertextType rhsCiphertextType =
      dyn_cast<lwe::LWECiphertextType>(rhs.getType());
  lwe::LWEPlaintextType lhsPlaintextType =
      dyn_cast<lwe::LWEPlaintextType>(lhs.getType());
  lwe::LWEPlaintextType rhsPlaintextType =
      dyn_cast<lwe::LWEPlaintextType>(rhs.getType());
  RankedTensorType lhsTensorType = dyn_cast<RankedTensorType>(lhs.getType());
  RankedTensorType rhsTensorType = dyn_cast<RankedTensorType>(rhs.getType());

  // Plaintext-Cleartext case
  if (lhsPlaintextType && rhsTensorType) {
    auto encodedLhs = cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
    return arith::MulFOp::create(builder, encodedLhs.getInput(), rhs)
        .getResult();
  }
  if (lhsTensorType && rhsPlaintextType) {
    auto encodedRhs = cast<lwe::RLWEEncodeOp>(rhs.getDefiningOp());
    return arith::MulFOp::create(builder, lhs, encodedRhs.getInput())
        .getResult();
  }

  // Plaintext-plaintext case
  if (lhsPlaintextType && rhsPlaintextType) {
    auto encodedLhs = cast<lwe::RLWEEncodeOp>(lhs.getDefiningOp());
    auto encodedRhs = cast<lwe::RLWEEncodeOp>(rhs.getDefiningOp());
    auto cleartextOp = arith::MulFOp::create(builder, encodedLhs.getInput(),
                                             encodedRhs.getInput());
    return lwe::RLWEEncodeOp::create(
        builder, lhsPlaintextType, cleartextOp.getResult(),
        lhsPlaintextType.getPlaintextSpace().getEncoding(),
        lhsPlaintextType.getPlaintextSpace().getRing());
  }

  // Ciphertext-Plaintext case
  if (lhsCiphertextType && rhsPlaintextType) {
    auto newRhs = encodeCleartextOperand(lhsCiphertextType, rhs,
                                         /*useDefaultScale=*/true);
    auto ctPtOp = ckks::MulPlainOp::create(builder, lhs, newRhs);
    return relinAndRescale(ctPtOp.getResult(),
                           /*relinearize=*/false,
                           /*rescale=*/rescaleAfterCtPtMul);
  }
  if (lhsPlaintextType && rhsCiphertextType) {
    auto newLhs = encodeCleartextOperand(rhsCiphertextType, lhs,
                                         /*useDefaultScale=*/true);
    auto ctPtOp = ckks::MulPlainOp::create(builder, newLhs, rhs);
    return relinAndRescale(ctPtOp.getResult(),
                           /*relinearize=*/false,
                           /*rescale=*/rescaleAfterCtPtMul);
  }

  // Ciphertext-Cleartext case
  if (lhsCiphertextType && rhsTensorType) {
    auto newRhs = encodeCleartextOperand(lhsCiphertextType, rhs,
                                         /*useDefaultScale=*/true);
    auto ctPtOp = ckks::MulPlainOp::create(builder, lhs, newRhs);
    return relinAndRescale(ctPtOp.getResult(),
                           /*relinearize=*/false,
                           /*rescale=*/rescaleAfterCtPtMul);
  }
  if (lhsTensorType && rhsCiphertextType) {
    auto newLhs = encodeCleartextOperand(rhsCiphertextType, lhs,
                                         /*useDefaultScale=*/true);
    auto ctPtOp = ckks::MulPlainOp::create(builder, newLhs, rhs);
    return relinAndRescale(ctPtOp.getResult(),
                           /*relinearize=*/false,
                           /*rescale=*/rescaleAfterCtPtMul);
  }

  // Ciphertext-ciphertext case
  assert(lhsCiphertextType && rhsCiphertextType);
  auto ctCtOp = ckks::MulOp::create(builder, lhs, rhs).getResult();
  return relinAndRescale(ctCtOp, /*relinearize=*/true,
                         /*rescale=*/true);
}

Value IRMaterializingVisitor::operator()(const LeftRotateNode<SSAValue>& node) {
  Value operand = this->process(node.operand);
  if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
    // The thing being rotated is a ciphertext-semantic tensor
    Value shift = arith::ConstantIntOp::create(builder, node.shift, 64);
    return tensor_ext::RotateOp::create(builder, operand.getType(), operand,
                                        shift);
  }

  IntegerAttr shift = builder.getI64IntegerAttr(node.shift);
  return ckks::RotateOp::create(builder, operand, shift);
}

Value IRMaterializingVisitor::operator()(const ExtractNode<SSAValue>& node) {
  Value operand = this->process(node.operand);
  RankedTensorType tensorType = cast<RankedTensorType>(operand.getType());
  Value index = arith::ConstantIndexOp::create(builder, node.index);
  if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(tensorType))) {
    return tensor::ExtractOp::create(builder, operand, index);
  }

  // Extracting 1 row of a matrix, so offset is 0 except for the row dim
  SmallVector<OpFoldResult> offsets(tensorType.getRank(),
                                    builder.getIndexAttr(0));
  offsets[0] = index;

  // Sizes are 1 in the row dim, and the full size in other dims
  SmallVector<OpFoldResult> sizes;
  sizes.push_back(builder.getIndexAttr(1));
  for (int i = 1; i < tensorType.getRank(); ++i) {
    sizes.push_back(builder.getIndexAttr(tensorType.getDimSize(i)));
  }

  // Strides are all 1
  SmallVector<OpFoldResult> strides(tensorType.getRank(),
                                    builder.getIndexAttr(1));

  RankedTensorType extractedType = RankedTensorType::get(
      tensorType.getShape().drop_front(), tensorType.getElementType());
  return builder.create<tensor::ExtractSliceOp>(extractedType, operand, offsets,
                                                sizes, strides);
}

}  // namespace orion
}  // namespace heir
}  // namespace mlir
