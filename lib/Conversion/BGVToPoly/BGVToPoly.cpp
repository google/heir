#include "include/Conversion/BGVToPoly/BGVToPoly.h"

#include <cstddef>
#include <optional>

#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/BGV/IR/BGVTypes.h"
#include "include/Dialect/Poly/IR/PolyAttributes.h"
#include "include/Dialect/Poly/IR/PolyTypes.h"
#include "include/Dialect/Poly/IR/Polynomial.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/FuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::bgv {

#define GEN_PASS_DEF_BGVTOPOLY
#include "include/Conversion/BGVToPoly/BGVToPoly.h.inc"

class CiphertextTypeConverter : public TypeConverter {
 public:
  // Convert ciphertext to tensor<#dim x !poly.poly<#rings[#level]>>
  CiphertextTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](CiphertextType type) -> Type {
      assert(type.getLevel().has_value());
      auto level = type.getLevel().value();
      assert(level < type.getRings().getRings().size());

      auto ring = type.getRings().getRings()[level];
      auto polyTy = poly::PolynomialType::get(ctx, ring);

      return RankedTensorType::get({type.getDim()}, polyTy);
    });
  }
  // We don't include any custom materialization ops because this lowering is
  // all done in a single pass. The dialect conversion framework works by
  // resolving intermediate (mid-pass) type conflicts by inserting
  // unrealized_conversion_cast ops, and only converting those to custom
  // materializations if they persist at the end of the pass. In our case,
  // we'd only need to use custom materializations if we split this lowering
  // across multiple passes.
};

struct BGVToPoly : public impl::BGVToPolyBase<BGVToPoly> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    CiphertextTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(context);

    // Add "standard" set of conversions and constraints for full dialect
    // conversion. See Dialect/Func/Transforms/FuncBufferize.cpp for example.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    // Run full conversion, if any BGV ops were missed out the pass will fail.
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::bgv
