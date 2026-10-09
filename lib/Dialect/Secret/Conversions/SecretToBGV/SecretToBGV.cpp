#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "lib/Dialect/BGV/IR/BGVAttributes.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVEnums.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtOps.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Secret/Conversions/Patterns.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Utils/AttributeUtils.h"
#include "lib/Utils/ContextAwareConversionUtils.h"
#include "lib/Utils/ContextAwareDialectConversion.h"
#include "lib/Utils/Polynomial/Polynomial.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

#define DEBUG_TYPE "secret-to-bgv"

namespace mlir::heir {

#define GEN_PASS_DEF_SECRETTOBGV
#include "lib/Dialect/Secret/Conversions/SecretToBGV/SecretToBGV.h.inc"

auto& kArgMgmtAttrName = mgmt::MgmtDialect::kArgMgmtAttrName;

class SecretToBGVTypeConverter : public SecretToRlweTypeConverter {
 public:
  SecretToBGVTypeConverter(MLIRContext* ctx, polynomial::RingAttr rlweRing,
                           int64_t ptm, bool isBFV)
      : SecretToRlweTypeConverter(ctx, rlweRing),
        plaintextModulus(ptm),
        isBFV(isBFV) {}

 protected:
  polynomial::RingAttr getPlaintextRing(MLIRContext* ctx) const override {
    return polynomial::RingAttr::get(
        ctx,
        mod_arith::ModArithType::get(
            ctx, IntegerAttr::get(IntegerType::get(ctx, 64), plaintextModulus)),
        ring.getPolynomialModulus());
  }

  Attribute getEncodingAttr(MLIRContext* ctx, int64_t scale) const override {
    return lwe::FullCRTPackingEncodingAttr::get(ctx, scale);
  }

  lwe::LweEncryptionType getEncryptionType() const override {
    return isBFV ? lwe::LweEncryptionType::msb : lwe::LweEncryptionType::lsb;
  }

 private:
  int64_t plaintextModulus;
  bool isBFV;
};

LogicalResult disallowFloatlike(const Type& type) {
  auto secretType = dyn_cast<secret::SecretType>(type);
  if (!secretType) return success();

  if (isa<FloatType>(getElementTypeOrSelf(secretType.getValueType())))
    return failure();

  return success();
}

struct SecretToBGV : public impl::SecretToBGVBase<SecretToBGV> {
  using SecretToBGVBase::SecretToBGVBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto* module = getOperation();

    auto schemeParamAttr = module->getAttrOfType<bgv::SchemeParamAttr>(
        bgv::BGVDialect::kSchemeParamAttrName);
    if (!schemeParamAttr) {
      module->emitError("expected BGV scheme parameters");
      signalPassFailure();
      return;
    }

    bool usePublicKey =
        schemeParamAttr.getEncryptionType() == bgv::BGVEncryptionType::pk;

    auto plaintextModulus = schemeParamAttr.getPlaintextModulus();
    auto rlweRing =
        lwe::getRlweRNSRing(context, schemeParamAttr.getQ().asArrayRef(),
                            1 << schemeParamAttr.getLogN());
    if (failed(rlweRing)) {
      return signalPassFailure();
    }
    // Ensure that all secret types are uniform and have last dimension
    // less than or equal to the ring parameter size. In other words, this
    // asserts that any data-semantic tensors have been converted to
    // ciphertext-semantic tensors with the correct shape.
    Operation* foundOp = walkAndDetect(module, [&](Operation* op) {
      ValueRange valuesToCheck = op->getOperands();
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        valuesToCheck = funcOp.getArguments();
      }
      for (auto value : valuesToCheck) {
        if (auto secretTy = dyn_cast<secret::SecretType>(value.getType())) {
          auto tensorTy = dyn_cast<RankedTensorType>(secretTy.getValueType());
          if (tensorTy && tensorTy.getDimSize(tensorTy.getRank() - 1) >
                              rlweRing.value()
                                  .getPolynomialModulus()
                                  .getPolynomial()
                                  .getDegree()) {
            return true;
          }
        }
      }
      return false;
    });
    if (foundOp != nullptr) {
      foundOp->emitError(
          "expected secret types to be tensors with last dimension "
          "less than or equal to ring parameter");
      signalPassFailure();
      return;
    }

    if (failed(walkAndValidateTypes<secret::GenericOp>(
            module, disallowFloatlike,
            "Floating point types are not supported in BGV. Maybe you meant "
            "to use a CKKS pipeline like --mlir-to-ckks?"))) {
      signalPassFailure();
      return;
    }

    // Invariant: for every SecretType, there is a corresponding MgmtAttr
    // attached to it, either in its DefiningOp or getOwner()->getParentOp()
    // (i.e., the FuncOp). Otherwise the typeConverter won't find the proper
    // type information and fail
    SecretToBGVTypeConverter typeConverter(context, rlweRing.value(),
                                           plaintextModulus,
                                           moduleIsBFV(getOperation()));

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    addSecretToSchemeDefaultConversionTargetsAndPatterns(patterns, target,
                                                         typeConverter);
    target.addLegalDialect<bgv::BGVDialect>();

    patterns.add<
        SecretGenericOpConversion<arith::AddIOp, bgv::AddOp>,
        SecretGenericOpConversion<arith::SubIOp, bgv::SubOp>,
        SecretGenericOpConversion<arith::MulIOp, bgv::MulOp>,
        SecretGenericOpRelinearizeConversion<bgv::RelinearizeOp>,
        SecretGenericOpModulusSwitchConversion<bgv::ModulusSwitchOp>,
        SecretGenericOpRotateConversion<bgv::RotateColumnsOp>,
        SecretGenericOpLevelReduceConversion<bgv::LevelReduceOp>,
        SecretGenericOpCipherPlainConversion<arith::AddIOp, bgv::AddPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::SubIOp, bgv::SubPlainOp>,
        SecretGenericOpCipherPlainConversion<arith::MulIOp, bgv::MulPlainOp>>(
        typeConverter, context);

    patterns.add<ConvertClientConceal>(typeConverter, context, usePublicKey,
                                       rlweRing.value());
    patterns.add<ConvertClientReveal>(typeConverter, context, rlweRing.value());

    if (failed(applyContextAwarePartialConversion(module, target,
                                                  std::move(patterns)))) {
      return signalPassFailure();
    }

    clearAttrs(getOperation(), mgmt::MgmtDialect::kArgMgmtAttrName);
    mgmt::cleanupInitOp(getOperation());
  }
};

}  // namespace mlir::heir
