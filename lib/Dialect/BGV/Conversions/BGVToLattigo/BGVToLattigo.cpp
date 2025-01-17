#include "lib/Dialect/BGV/Conversions/BGVToLattigo/BGVToLattigo.h"

#include <cassert>
#include <utility>
#include <vector>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/Conversions/RlweToLattigo/RlweToLattigo.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "lib/Utils/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOLATTIGO
#include "lib/Dialect/BGV/Conversions/BGVToLattigo/BGVToLattigo.h.inc"

using ConvertAddOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RAddOp, lattigo::BGVAddOp>;
using ConvertSubOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RSubOp, lattigo::BGVSubOp>;
using ConvertMulOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, lwe::RMulOp, lattigo::BGVMulOp>;
using ConvertAddPlainOp = ConvertRlwePlainOp<lattigo::BGVEvaluatorType,
                                             AddPlainOp, lattigo::BGVAddOp>;
using ConvertSubPlainOp = ConvertRlwePlainOp<lattigo::BGVEvaluatorType,
                                             SubPlainOp, lattigo::BGVSubOp>;
using ConvertMulPlainOp = ConvertRlwePlainOp<lattigo::BGVEvaluatorType,
                                             MulPlainOp, lattigo::BGVMulOp>;

using ConvertRelinOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, RelinearizeOp,
                       lattigo::BGVRelinearizeOp>;
using ConvertModulusSwitchOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, ModulusSwitchOp,
                       lattigo::BGVRescaleOp>;

// TODO(#1186): figure out generic rotating using BGVRotateColumns/RowsOp
using ConvertRotateOp = ConvertRlweRotateOp<lattigo::BGVEvaluatorType, RotateOp,
                                            lattigo::BGVRotateColumnsOp>;

using ConvertEncryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEEncryptorType, lwe::RLWEEncryptOp,
                       lattigo::RLWEEncryptOp>;
using ConvertDecryptOp =
    ConvertRlweUnaryOp<lattigo::RLWEDecryptorType, lwe::RLWEDecryptOp,
                       lattigo::RLWEDecryptOp>;
using ConvertEncodeOp =
    ConvertRlweEncodeOp<lattigo::BGVEncoderType, lattigo::BGVParameterType,
                        lwe::RLWEEncodeOp, lattigo::BGVEncodeOp,
                        lattigo::BGVNewPlaintextOp>;
using ConvertDecodeOp =
    ConvertRlweDecodeOp<lattigo::BGVEncoderType, lwe::RLWEDecodeOp,
                        lattigo::BGVDecodeOp, arith::ConstantOp>;

struct ConvertLWEReinterpretUnderlyingType
    : public OpConversionPattern<lwe::ReinterpretUnderlyingTypeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      lwe::ReinterpretUnderlyingTypeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // erase reinterpret underlying
    rewriter.replaceOp(op, adaptor.getOperands()[0].getDefiningOp());
    return success();
  }
};

struct BGVToLattigo : public impl::BGVToLattigoBase<BGVToLattigo> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToLattigoTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<lattigo::LattigoDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();
    target
        .addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp, lwe::RLWEEncodeOp,
                      lwe::RLWEDecodeOp, lwe::RAddOp, lwe::RSubOp, lwe::RMulOp,
                      lwe::ReinterpretUnderlyingTypeOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg =
          op.getFunctionType().getNumInputs() > 0 &&
          containsArgumentOfType<
              lattigo::BGVEvaluatorType, lattigo::BGVEncoderType,
              lattigo::RLWEEncryptorType, lattigo::RLWEDecryptorType>(op);

      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsDialects<lwe::LWEDialect, bgv::BGVDialect>(op) ||
              hasCryptoContextArg);
    });

    std::vector<std::pair<Type, OpPredicate>> evaluators;

    // param/encoder also needed for the main func
    // as there might (not) be ct-pt operations
    evaluators = {
        {lattigo::BGVEvaluatorType::get(context),
         containsDialects<lwe::LWEDialect, bgv::BGVDialect>},
        {lattigo::BGVParameterType::get(context),
         containsDialects<lwe::LWEDialect, bgv::BGVDialect>},
        {lattigo::BGVEncoderType::get(context),
         containsDialects<lwe::LWEDialect, bgv::BGVDialect>},
        {lattigo::RLWEEncryptorType::get(context),
         containsAnyOperations<lwe::RLWEEncryptOp>},
        {lattigo::RLWEDecryptorType::get(context),
         containsAnyOperations<lwe::RLWEDecryptOp>},
    };

    patterns.add<AddEvaluatorArg>(context, evaluators);

    patterns.add<ConvertAddOp, ConvertSubOp, ConvertMulOp, ConvertAddPlainOp,
                 ConvertSubPlainOp, ConvertMulPlainOp, ConvertRelinOp,
                 ConvertModulusSwitchOp, ConvertRotateOp, ConvertEncryptOp,
                 ConvertDecryptOp, ConvertEncodeOp, ConvertDecodeOp,
                 ConvertLWEReinterpretUnderlyingType>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }

    // remove unused key args from function types
    // in favor of encryptor/decryptor
    RewritePatternSet postPatterns(context);
    postPatterns.add<RemoveKeyArg<lattigo::RLWESecretKeyType>>(context);
    postPatterns.add<RemoveKeyArg<lattigo::RLWEPublicKeyType>>(context);
    walkAndApplyPatterns(module, std::move(postPatterns));
  }
};

}  // namespace mlir::heir::bgv
