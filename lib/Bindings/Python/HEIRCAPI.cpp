#include "lib/Bindings/Python/HEIRCAPI.h"

#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/Transforms/Passes.h"
#include "mlir/include/mlir/CAPI/IR.h"                   // from @llvm-project
#include "mlir/include/mlir/CAPI/Registration.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"         // from @llvm-project

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Secret, secret,
                                      mlir::heir::secret::SecretDialect)

void heirRegisterAllDialects(MlirDialectRegistry registry) {
  mlir::DialectRegistry* r = unwrap(registry);
  r->insert<mlir::heir::secret::SecretDialect, mlir::arith::ArithDialect,
            mlir::func::FuncDialect, mlir::tensor::TensorDialect>();
}

void heirRegisterAllPasses(void) {
  mlir::registerTransformsPasses();
  mlir::heir::secret::registerSecretPasses();
}
