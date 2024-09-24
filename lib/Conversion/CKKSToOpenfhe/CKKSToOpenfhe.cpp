#include "lib/Conversion/CKKSToOpenfhe/CKKSToOpenfhe.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "lib/Conversion/LWEToOpenfhe/LWEToOpenfhe.h"
#include "lib/Conversion/RlweToOpenfhe/RlweToOpenfhe.h"
#include "lib/Conversion/Utils.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::ckks {

#define GEN_PASS_DEF_CKKSTOOPENFHE
#include "lib/Conversion/CKKSToOpenfhe/CKKSToOpenfhe.h.inc"

using ConvertNegateOp = ConvertRlweUnaryOp<NegateOp, openfhe::NegateOp>;
using ConvertAddOp = ConvertRlweBinOp<AddOp, openfhe::AddOp>;
using ConvertSubOp = ConvertRlweBinOp<SubOp, openfhe::SubOp>;
using ConvertMulOp = ConvertRlweBinOp<MulOp, openfhe::MulNoRelinOp>;
using ConvertAddPlainOp =
    ConvertRlweCiphertextPlaintextOp<AddPlainOp, openfhe::AddPlainOp>;
using ConvertMulPlainOp =
    ConvertRlweCiphertextPlaintextOp<MulPlainOp, openfhe::MulPlainOp>;
using ConvertRotateOp = ConvertRlweRotateOp<RotateOp>;
using ConvertRelinOp = ConvertRlweRelinOp<RelinearizeOp>;
using ConvertExtractOp =
    lwe::ConvertRlweExtractOp<ExtractOp, MulPlainOp, RotateOp>;

struct CKKSToOpenfhe : public impl::CKKSToOpenfheBase<CKKSToOpenfhe> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToOpenfheTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<openfhe::OpenfheDialect>();
    target.addIllegalDialect<ckks::CKKSDialect>();
    target.addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp,
                        lwe::RLWEEncodeOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg = op.getFunctionType().getNumInputs() > 0 &&
                                 mlir::isa<openfhe::CryptoContextType>(
                                     *op.getFunctionType().getInputs().begin());
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsLweOrDialect<ckks::CKKSDialect>(op) ||
              hasCryptoContextArg);
    });

    patterns
        .add<AddCryptoContextArg<ckks::CKKSDialect>, ConvertAddOp, ConvertSubOp,
             ConvertMulOp, ConvertAddPlainOp, ConvertMulPlainOp,
             ConvertNegateOp, ConvertRotateOp, ConvertRelinOp, ConvertExtractOp,
             lwe::ConvertEncryptOp, lwe::ConvertDecryptOp>(typeConverter,
                                                           context);
    patterns.add<lwe::ConvertEncodeOp>(typeConverter, context, /*ckks=*/true);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::ckks
