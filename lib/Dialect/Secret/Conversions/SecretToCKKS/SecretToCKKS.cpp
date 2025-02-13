#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOCKKS
#include "lib/Dialect/Secret/Conversions/SecretToCKKS/SecretToCKKS.h.inc"

namespace {

// Returns an RLWE ring given the specified number of bits needed and polynomial
// modulus degree.
// TODO(#536): Integrate a general library to compute appropriate prime moduli
// given any number of bits.
FailureOr<polynomial::RingAttr> getRlweRNSRing(MLIRContext *ctx,
                                               int currentLevel,
                                               int coefficientModBits,
                                               int polyModDegree) {
  std::vector<::mlir::heir::polynomial::IntMonomial> monomials;
  monomials.emplace_back(1, polyModDegree);
  monomials.emplace_back(1, 0);
  auto result =
      ::mlir::heir::polynomial::IntPolynomial::fromMonomials(monomials);
  if (failed(result)) return failure();
  ::mlir::heir::polynomial::IntPolynomial xnPlusOne = result.value();
  // all 40 bit primes...
  std::vector<int64_t> primes = {1095233372161, 1032955396097, 1005037682689,
                                 998595133441,  972824936449,  959939837953};
  SmallVector<Type, 4> modTypes;
  for (int i = 0; i <= currentLevel; i++) {
    auto type = IntegerType::get(ctx, 64);
    modTypes.push_back(
        mod_arith::ModArithType::get(ctx, IntegerAttr::get(type, primes[i])));
  }
  auto rnsType = rns::RNSType::get(ctx, modTypes);
  return ::mlir::heir::polynomial::RingAttr::get(
      rnsType, polynomial::IntPolynomialAttr::get(ctx, xnPlusOne));
}

polynomial::RingAttr getRlweRNSRingWithLevel(polynomial::RingAttr ringAttr,
                                             int level) {
  auto rnsType = cast<rns::RNSType>(ringAttr.getCoefficientType());

  auto newRnsType = rns::RNSType::get(
      rnsType.getContext(), rnsType.getBasisTypes().take_front(level + 1));
  return ::mlir::heir::polynomial::RingAttr::get(
      newRnsType, ringAttr.getPolynomialModulus());
}

// Returns the unique non-unit dimension of a tensor and its rank.
// Returns failure if the tensor has more than one non-unit dimension.
FailureOr<std::pair<unsigned, int64_t>> getNonUnitDimension(
    RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();

  if (llvm::count_if(shape, [](auto dim) { return dim != 1; }) != 1) {
    return failure();
  }

  unsigned nonUnitIndex = std::distance(
      shape.begin(), llvm::find_if(shape, [&](auto dim) { return dim != 1; }));

  return std::make_pair(nonUnitIndex, shape[nonUnitIndex]);
}

}  // namespace

class SecretToCKKSTypeConverter : public TypeWithAttrTypeConverter {
 public:
  SecretToCKKSTypeConverter(MLIRContext *ctx,
                            ::mlir::heir::polynomial::RingAttr rlweRing,
                            bool packTensorInSlots)
      : TypeWithAttrTypeConverter(mgmt::MgmtDialect::kArgMgmtAttrName) {
    addConversion([](Type type) { return type; });

    ring_ = rlweRing;
    packTensorInSlots_ = packTensorInSlots;
  }

  Type convertSecretTypeWithMgmtAttr(secret::SecretType type,
                                     mgmt::MgmtAttr mgmtAttr) const {
    auto *ctx = type.getContext();
    auto level = mgmtAttr.getLevel();
    auto dimension = mgmtAttr.getDimension();

    Type valueTy = type.getValueType();

    // Note that slot number for CKKS is always half of the ring dimension.
    // so ring_.getPolynomialModulus() is not useful here
    // TODO(#1191): use packing information to get the correct slot number
    auto plaintextRing = ::mlir::heir::polynomial::RingAttr::get(
        type.getContext(), Float64Type::get(ctx), ring_.getPolynomialModulus());

    SmallVector<IntegerAttr, 6> moduliChain;
    for (auto modArithType :
         cast<rns::RNSType>(ring_.getCoefficientType()).getBasisTypes()) {
      auto modulus = cast<mod_arith::ModArithType>(modArithType).getModulus();
      moduliChain.push_back(modulus);
    }

    // TODO(#785): Set a scaling parameter for floating point values.
    auto ciphertext = lwe::NewLWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, type.getValueType(),
                                      lwe::NoOverflowAttr::get(ctx)),
        lwe::PlaintextSpaceAttr::get(
            ctx, plaintextRing,
            lwe::InverseCanonicalEncodingAttr::get(ctx, 1 << 10)),
        lwe::CiphertextSpaceAttr::get(ctx,
                                      getRlweRNSRingWithLevel(ring_, level),
                                      lwe::LweEncryptionType::lsb, dimension),
        lwe::KeyAttr::get(ctx, 0),
        lwe::ModulusChainAttr::get(ctx, moduliChain, level));

    // Return a single ciphertext if inputs are packed into a single
    // ciphertext SIMD slot or the secret value type is a scalar.
    if (this->packTensorInSlots_ || !isa<TensorType>(valueTy)) {
      return ciphertext;
    }
    // If the input IR does not contain aligned ciphertexts, we will not
    // pack tensors into ciphertext SIMD slots, so tensors are converted
    // into tensors of RLWE ciphertexts.
    assert(dyn_cast<RankedTensorType>(valueTy) &&
           "expected ranked tensor type");
    auto scalarType = cast<RankedTensorType>(valueTy).getElementType();
    ciphertext = lwe::NewLWECiphertextType::get(
        ctx,
        lwe::ApplicationDataAttr::get(ctx, scalarType,
                                      lwe::NoOverflowAttr::get(ctx)),
        ciphertext.getPlaintextSpace(), ciphertext.getCiphertextSpace(),
        ciphertext.getKey(), ciphertext.getModulusChain());
    return RankedTensorType::get(cast<RankedTensorType>(valueTy).getShape(),
                                 ciphertext);
  }

  Type convertTypeWithAttr(Type type, Attribute attr) const override {
    auto secretTy = dyn_cast<secret::SecretType>(type);
    // guard against null attribute
    if (secretTy && attr) {
      auto mgmtAttr = dyn_cast<mgmt::MgmtAttr>(attr);
      if (mgmtAttr) {
        return convertSecretTypeWithMgmtAttr(secretTy, mgmtAttr);
      }
    }
    return type;
  }

 private:
  ::mlir::heir::polynomial::RingAttr ring_;
  bool packTensorInSlots_;
};

class SecretGenericTensorExtractConversion
    : public SecretGenericOpConversion<tensor::ExtractOp, ckks::ExtractOp> {
 public:
  using SecretGenericOpConversion<tensor::ExtractOp,
                                  ckks::ExtractOp>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    auto inputTy = inputs[0].getType();
    if (!isa<lwe::NewLWECiphertextType>(getElementTypeOrSelf(inputTy))) {
      return failure();
    }
    if (isa<RankedTensorType>(inputTy)) {
      // TODO(#1174): decide this in earlier pipeline
      // Extracts an element out of a tensor (the secret tensor is not
      // packed).
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, outputTypes, inputs);
      return success();
    }
    // Extracts an element out of a slot of a single ciphertext.
    // TODO(#913): Once we have a layout descriptor, we should be able to
    // translate a tensor.extract into the appropriate ckks.extract operation.
    // For now, if there we are extracting a multi-dimensional tensor with
    // only one non-unit dimension stored in a single ciphertext along that
    // dimension, then extract on the index of the non-unit dimension.
    auto lweCiphertextInputTy = cast<lwe::NewLWECiphertextType>(inputTy);
    auto underlyingTy = cast<RankedTensorType>(
        lweCiphertextInputTy.getApplicationData().getMessageType());
    auto nonUnitDim = getNonUnitDimension(underlyingTy);
    if (failed(nonUnitDim)) {
      return failure();
    }
    assert(inputs.size() == 1 + underlyingTy.getRank() &&
           "expected tensor.extract inputs for each index");
    auto nonUnitShift = inputs[1 + nonUnitDim.value().first];
    rewriter.replaceOpWithNewOp<ckks::ExtractOp>(op, outputTypes[0], inputs[0],
                                                 nonUnitShift);
    return success();
  }
};

class SecretGenericTensorInsertConversion
    : public SecretGenericOpConversion<tensor::InsertOp, tensor::InsertOp> {
 public:
  using SecretGenericOpConversion<tensor::InsertOp,
                                  tensor::InsertOp>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<lwe::NewLWECiphertextType>(inputs[0].getType())) {
      op.emitError()
          << "expected scalar to insert to be of type RLWE ciphertext"
          << inputs[0].getType();
      return failure();
    }
    if (isa<RankedTensorType>(inputs[1].getType())) {
      // Insert an element into a tensor (the secret tensor is not packed).
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, outputTypes, inputs);
      return success();
    }
    // We can also support the case where the secret tensor is packed into a
    // single ciphertext by converting the insert operation into a zero-hot
    // multiplication followed by an addition of the scalar encoded into a
    // plaintext in the correct slot.
    return failure();
  }
};

class SecretForOpConversion : public OpConversionPattern<affine::AffineForOp> {
 public:
  using OpConversionPattern<affine::AffineForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      affine::AffineForOp forOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    const auto *contextAwareTypeConverter =
        dynamic_cast<const TypeWithAttrTypeConverter *>(getTypeConverter());
    SmallVector<Type> newInitTypes;
    contextAwareTypeConverter->convertValueRangeTypes(forOp.getInits(),
                                                      newInitTypes);

    Location loc = forOp.getLoc();
    auto newForOp = rewriter.create<affine::AffineForOp>(
        loc, ValueRange(forOp.getLowerBoundOperands()),
        forOp.getLowerBoundMap(), ValueRange(forOp.getUpperBoundOperands()),
        forOp.getUpperBoundMap(), forOp.getStep().getZExtValue(),
        adaptor.getInits(), [](OpBuilder &, Location, Value, ValueRange) {});
    newForOp->setAttrs(forOp->getAttrs());

    mlir::Block *newBody = newForOp.getBody();
    mlir::Block *oldBody = forOp.getBody();
    rewriter.setInsertionPoint(newBody, newBody->begin());

    SmallVector<Value> newBlockArgs;
    for (auto arg : newBody->getArguments()) {
      if (auto lweTy = dyn_cast<lwe::NewLWECiphertextType>(arg.getType())) {
        // For each block arg that is a secret type, we need to create an
        // operation inside the for loops block that can hold the mgmt attr for
        // type conversion. We can't rely on getValueAttr since blocks are
        // unlinked during dialect conversion.
        auto underlyingTy = secret::SecretType::get(
            lweTy.getApplicationData().getMessageType());
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
            loc, underlyingTy, arg);
        cast->setAttrs(oldBody->getArgument(arg.getArgNumber())
                           .getUsers()
                           .begin()
                           ->getAttrs());
        newBlockArgs.push_back(cast.getResult(0));
      } else {
        newBlockArgs.push_back(arg);
      }
    }

    // Move the body of the old ForOp to the new one.
    rewriter.mergeBlocks(oldBody, newBody, newBlockArgs);
    rewriter.replaceOp(forOp, newForOp);
    return success();
  }
};

class SecretGenericFuncCallConversion
    : public SecretGenericOpConversion<func::CallOp, func::CallOp> {
 public:
  using SecretGenericOpConversion<func::CallOp,
                                  func::CallOp>::SecretGenericOpConversion;

  LogicalResult matchAndRewriteInner(
      secret::GenericOp op, TypeRange outputTypes, ValueRange inputs,
      ArrayRef<NamedAttribute> attributes,
      ConversionPatternRewriter &rewriter) const override {
    // check if any args are secret from wrapping generic
    // clone the callee (and update a unique name, for now always) the call
    // operands add a note that we don't have to always clone to be secret
    // update the called func's type signature

    func::CallOp callOp = *op.getBody()->getOps<func::CallOp>().begin();
    auto module = callOp->getParentOfType<ModuleOp>();
    func::FuncOp callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());

    // For now, ensure that there is only one caller to the callee.
    auto calleeUses = callee.getSymbolUses(callee->getParentOp());
    if (std::distance(calleeUses->begin(), calleeUses->end()) != 1) {
      return op->emitError() << "expected exactly one caller to the callee";
    }

    SmallVector<Type> newInputTypes;
    for (auto val : inputs) {
      newInputTypes.push_back(val.getType());
    }

    FunctionType newFunctionType =
        cast<FunctionType>(callee.cloneTypeWith(newInputTypes, outputTypes));
    auto newFuncOp = rewriter.cloneWithoutRegions(callee);
    newFuncOp->moveAfter(callee);
    newFuncOp.setFunctionType(newFunctionType);
    newFuncOp.setSymName(
        llvm::formatv("{0}_secret", callee.getSymName()).str());

    auto newCallOp = rewriter.create<func::CallOp>(op.getLoc(), outputTypes,
                                                   newFuncOp.getName(), inputs);
    rewriter.replaceOp(op, newCallOp);
    rewriter.eraseOp(callee);
    return success();
  }
};

struct SecretToCKKS : public impl::SecretToCKKSBase<SecretToCKKS> {
  using SecretToCKKSBase::SecretToCKKSBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    // Helper for future lowerings that want to know what scheme was used
    module->setAttr(kCKKSSchemeAttrName, UnitAttr::get(context));

    auto maxLevel = 5;
    auto rlweRing =
        getRlweRNSRing(context, maxLevel, coefficientModBits, polyModDegree);
    if (failed(rlweRing)) {
      return signalPassFailure();
    }
    // Ensure that all secret types are uniform and matching the ring
    // parameter size in order to pack tensors into ciphertext SIMD slots.
    // TODO(#1174): decide this earlier, remove polyModDegree param to earlier
    // pipeline
    LogicalResult validationResult =
        walkAndValidateValues(module, [&](Value value) {
          if (auto secretTy = dyn_cast<secret::SecretType>(value.getType())) {
            auto tensorTy = dyn_cast<RankedTensorType>(secretTy.getValueType());
            if (tensorTy) {
              // TODO(#913): Multidimensional tensors with a single non-unit
              // dimension are assumed to be packed in the order of that
              // dimensions.
              auto nonUnitDim = getNonUnitDimension(tensorTy);
              if (failed(nonUnitDim) ||
                  nonUnitDim.value().second != polyModDegree) {
                return failure();
              }
            }
          }
          return success();
        });
    if (failed(validationResult)) {
      emitWarning(module->getLoc(),
                  "expected secret types to be tensors with dimension matching "
                  "ring parameter, pass will not pack tensors into ciphertext "
                  "SIMD slots");
    }
    bool packTensorInSlots = succeeded(validationResult);

    // Invariant: for every SecretType, there is a
    // corresponding MgmtAttr attached to it,
    // either in its DefiningOp or getOwner()->getParentOp()
    // (i.e., the FuncOp).
    // Otherwise the typeConverter won't find the proper type information
    // and fail
    SecretToCKKSTypeConverter typeConverter(context, rlweRing.value(),
                                            packTensorInSlots);
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<ckks::CKKSDialect, lwe::LWEDialect>();
    target.addIllegalDialect<secret::SecretDialect>();
    target.addIllegalOp<secret::GenericOp>();
    target.addIllegalOp<mgmt::ModReduceOp, mgmt::RelinearizeOp>();
    // for mod reduce on tensor ciphertext
    target.addLegalOp<arith::ConstantOp, tensor::EmptyOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isFuncArgumentAndResultLegal(op);
    });
    // to resolve unlinked block arguments
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();

    // We add an explicit allowlist of operations to mark legal. If we use
    // markUnknownOpDynamicallyLegal, then ConvertAny will be applied to any
    // remaining operations and potentially cause a crash.
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](Operation *op) { return typeConverter.isOperationLegal(op); });
    target.addDynamicallyLegalOp<affine::AffineYieldOp>(
        [&](Operation *op) { return typeConverter.isOperationLegal(op); });
    target.addDynamicallyLegalOp<tensor::ExtractOp, tensor::InsertOp>(
        [&](Operation *op) { return typeConverter.isOperationLegal(op); });
    // We can't use typeConverter.isOperationLegal here because it requires the
    // region being converted.
    target.addDynamicallyLegalOp<affine::AffineForOp>(
        [&](affine::AffineForOp op) {
          for (auto operand : op->getOperands()) {
            if (!typeConverter.isValueLegal(operand)) {
              return false;
            }
          }
          for (auto result : op->getResults()) {
            if (!typeConverter.isValueLegal(result)) {
              return false;
            }
          }
          return true;
        });

    patterns.add<
        ConvertFuncWithContextAwareTypeConverter,
        SecretGenericOpCipherConversion<arith::AddIOp, ckks::AddOp>,
        SecretGenericOpCipherConversion<arith::SubIOp, ckks::SubOp>,
        SecretGenericOpCipherConversion<arith::MulIOp, ckks::MulOp>,
        SecretGenericOpCipherConversion<arith::AddFOp, ckks::AddOp>,
        SecretGenericOpCipherConversion<arith::SubFOp, ckks::SubOp>,
        SecretGenericOpCipherConversion<arith::MulFOp, ckks::MulOp>,
        SecretGenericOpCipherConversion<tensor::EmptyOp, tensor::EmptyOp>,
        SecretGenericOpRelinearizeConversion<ckks::RelinearizeOp>,
        SecretGenericOpModulusSwitchConversion<ckks::RescaleOp>,
        SecretGenericOpCipherConversion<mgmt::BootstrapOp, ckks::BootstrapOp>,
        SecretGenericTensorExtractConversion,
        SecretGenericTensorInsertConversion,
        SecretGenericOpRotateConversion<ckks::RotateOp>,
        SecretGenericOpCipherPlainConversion<arith::AddFOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubFOp, ckks::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulFOp, ckks::MulPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, ckks::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, ckks::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, ckks::MulPlainOp>,
        SecretForOpConversion, ConvertAny<affine::AffineYieldOp>,
        SecretGenericFuncCallConversion>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // cleanup MgmtAttr
    SmallVector<Operation *> opsToRemove;
    getOperation()->walk([&](Operation *op) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        for (auto i = 0; i != funcOp.getNumArguments(); ++i) {
          funcOp.removeArgAttr(i, mgmt::MgmtDialect::kArgMgmtAttrName);
        }
      } else {
        op->removeAttr(mgmt::MgmtDialect::kArgMgmtAttrName);
      }
      if (auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
        if (castOp.getOperand(0).getType() == castOp.getResult(0).getType()) {
          castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
          opsToRemove.push_back(castOp);
        }
      }
    });
    for (auto op : opsToRemove) {
      op->erase();
    }
  }
};

}  // namespace mlir::heir
