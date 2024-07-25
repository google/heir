#include "lib/Conversion/LWEToPolynomial/LWEToPolynomial.h"

#include <cstddef>
#include <optional>
#include <utility>

#include "lib/Conversion/Utils.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/Polynomial.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOPOLYNOMIAL
#include "lib/Conversion/LWEToPolynomial/LWEToPolynomial.h.inc"

class CiphertextTypeConverter : public TypeConverter {
 public:
  // Convert ciphertext to tensor<#dim x !poly.poly<#rings[#level]>>
  // Our precondition is that the key and plaintext have only one
  // dimension, and the ciphertext has two dimensions. (i.e.) The key is a
  // single polynomial and not a tensor of higher dimensions. We may support
  // higher dimensions in the future (for schemes such as TFHE).
  CiphertextTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](lwe::RLWECiphertextType type) -> Type {
      auto rlweParams = type.getRlweParams();
      auto ring = rlweParams.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({rlweParams.getDimension()}, polyTy);
    });
    addConversion([ctx](lwe::RLWEPlaintextType type) -> Type {
      auto ring = type.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);
      return polyTy;
    });
    addConversion([ctx](lwe::RLWESecretKeyType type) -> Type {
      auto rlweParams = type.getRlweParams();
      auto ring = rlweParams.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({rlweParams.getDimension() - 1}, polyTy);
    });
    addConversion([ctx](lwe::RLWEPublicKeyType type) -> Type {
      auto rlweParams = type.getRlweParams();
      auto ring = rlweParams.getRing();
      auto polyTy = ::mlir::polynomial::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({rlweParams.getDimension()}, polyTy);
    });
  }
};

struct ConvertRLWEDecrypt : public OpConversionPattern<RLWEDecryptOp> {
  ConvertRLWEDecrypt(mlir::MLIRContext *context)
      : OpConversionPattern<RLWEDecryptOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RLWEDecryptOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto secretKey = adaptor.getSecretKey();

    auto inputDimension = cast<RankedTensorType>(input.getType()).getShape();

    // Note that secretKey dimension needs to be one. The following check is
    // sufficient for this because before this is executed, both the input
    // (a ciphertext) and secretKey (a secret key) are checked to have the same
    // RLWE dimensions; otherwise, it errors out in advance.
    if (inputDimension.size() != 1 || inputDimension.front() != 2) {
      op.emitError() << "Expected 2 dimensional ciphertext, found ciphertext "
                        "tensor dimension = "
                     << inputDimension;
      // TODO: For TFHE, which can support higher dimensional keys, plaintexts,
      // and ciphertexts, we need to add support for encrypt and decrypt for
      // those cases.
      return failure();
    }

    ImplicitLocOpBuilder builder(loc, rewriter);

    // For a ciphertext input = (c_0, c_1), calculates
    // plaintext = secretKey * c_0 + c_1
    auto index0 = builder.create<arith::ConstantIndexOp>(0);
    tensor::ExtractOp extractOp0 =
        builder.create<tensor::ExtractOp>(input, ValueRange{index0});
    auto index1 = builder.create<arith::ConstantIndexOp>(1);
    tensor::ExtractOp extractOp1 =
        builder.create<tensor::ExtractOp>(input, ValueRange{index1});

    tensor::ExtractOp extractSecretKeyOp =
        builder.create<tensor::ExtractOp>(secretKey, ValueRange{index0});
    auto index1sk =
        builder.create<polynomial::MulOp>(extractSecretKeyOp, extractOp0);
    auto plaintext = builder.create<polynomial::AddOp>(index1sk, extractOp1);

    rewriter.replaceOp(op, plaintext);
    return success();
  }
};

struct LWEToPolynomial : public impl::LWEToPolynomialBase<LWEToPolynomial> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    CiphertextTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(context);

    patterns.add<ConvertRLWEDecrypt>(typeConverter, context);
    target.addIllegalOp<RLWEDecryptOp>();

    addStructuralConversionPatterns(typeConverter, patterns, target);

    // Run full conversion, if any LWE ops were missed out the pass will fail.
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::lwe
