#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h"

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "lib/Dialect/Cheddar/IR/CheddarDialect.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project

namespace mlir::heir::cheddar {

#define GEN_PASS_DEF_CONFIGURECRYPTOCONTEXT
#include "lib/Dialect/Cheddar/Transforms/ConfigureCryptoContext.h.inc"

struct ConfigureCryptoContext
    : public impl::ConfigureCryptoContextBase<ConfigureCryptoContext> {
  using ConfigureCryptoContextBase::ConfigureCryptoContextBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    auto schemeParamAttr = moduleOp->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);

    if (schemeParamAttr) {
      int64_t logN = schemeParamAttr.getLogN();
      int64_t logDefaultScale = schemeParamAttr.getLogDefaultScale();

      moduleOp->setAttr("cheddar.logN",
                        IntegerAttr::get(IntegerType::get(ctx, 64), logN));
      moduleOp->setAttr(
          "cheddar.logDefaultScale",
          IntegerAttr::get(IntegerType::get(ctx, 64), logDefaultScale));

      if (auto Q = schemeParamAttr.getQ()) {
        moduleOp->setAttr("cheddar.Q", Q);
      }

      if (auto P = schemeParamAttr.getP()) {
        moduleOp->setAttr("cheddar.P", P);
      }

      moduleOp->removeAttr(ckks::CKKSDialect::kSchemeParamAttrName);
    }

    moduleOp->removeAttr("scheme.ckks");
  }
};

}  // namespace mlir::heir::cheddar
