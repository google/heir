#include "include/Conversion/CombToCGGI/CombToCGGI.h"

#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/CGGI/IR/CGGIOps.h"
#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Dialect/Comb/IR/CombOps.h"
#include "include/Dialect/LWE/IR/LWETypes.h"
#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Conversion/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::comb {

#define GEN_PASS_DEF_COMBTOCGGI
#include "include/Conversion/CombToCGGI/CombToCGGI.h.inc"

class SecretTypeConverter : public TypeConverter {
 public:
  SecretTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });

    // Convert secret types to LWE ciphertext types
    addConversion([ctx](secret::SecretType type) -> Type {
      auto intType = dyn_cast<IntegerType>(type.getValueType());
      assert(intType);
      return lwe::LWECiphertextType::get(
          ctx,
          lwe::UnspecifiedBitFieldEncodingAttr::get(ctx, intType.getWidth()),
          lwe::LWEParamsAttr());
    });
  }
};

struct CombToCGGI : public impl::CombToCGGIBase<CombToCGGI> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    SecretTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(context);
    addStructuralConversionPatterns(typeConverter, patterns, target);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace mlir::heir::comb
