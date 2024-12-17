#include "lib/Dialect/BGV/Conversions/BGVToLattigo/BGVToLattigo.h"

#include <cassert>
#include <utility>

#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/BGV/IR/BGVOps.h"
#include "lib/Dialect/LWE/Conversions/RlweToLattigo/RlweToLattigo.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Utils/ConversionUtils/ConversionUtils.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOLATTIGO
#include "lib/Dialect/BGV/Conversions/BGVToLattigo/BGVToLattigo.h.inc"

using ConvertAddOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, AddOp, lattigo::BGVAddOp>;
using ConvertSubOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, SubOp, lattigo::BGVSubOp>;
using ConvertMulOp =
    ConvertRlweBinOp<lattigo::BGVEvaluatorType, MulOp, lattigo::BGVMulOp>;
using ConvertRelinOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, RelinearizeOp,
                       lattigo::BGVRelinearizeOp>;
using ConvertModulusSwitchOp =
    ConvertRlweUnaryOp<lattigo::BGVEvaluatorType, ModulusSwitchOp,
                       lattigo::BGVRescaleOp>;

// TODO(#1186): figure out generic rotating using BGVRotateColumns/RowsOp
using ConvertRotateOp = ConvertRlweRotateOp<lattigo::BGVEvaluatorType, RotateOp,
                                            lattigo::BGVRotateColumnsOp>;

struct BGVToLattigo : public impl::BGVToLattigoBase<BGVToLattigo> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    ToLattigoTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<lattigo::LattigoDialect>();
    target.addIllegalDialect<bgv::BGVDialect>();
    target.addIllegalOp<lwe::RLWEEncryptOp, lwe::RLWEDecryptOp,
                        lwe::RLWEEncodeOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      bool hasCryptoContextArg = op.getFunctionType().getNumInputs() > 0 &&
                                 mlir::isa<lattigo::BGVEvaluatorType>(
                                     *op.getFunctionType().getInputs().begin());
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody()) &&
             (!containsDialects<lwe::LWEDialect, bgv::BGVDialect>(op) ||
              hasCryptoContextArg);
    });

    patterns.add<AddEvaluatorArg<bgv::BGVDialect, lattigo::BGVEvaluatorType>,
                 ConvertAddOp, ConvertSubOp, ConvertMulOp, ConvertRelinOp,
                 ConvertModulusSwitchOp, ConvertRotateOp>(typeConverter,
                                                          context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::bgv
