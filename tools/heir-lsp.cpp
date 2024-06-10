#include "lib/Dialect/ArithExt/IR/ArithExtDialect.h"
#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/PolyExt/IR/PolyExtDialect.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h"          // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project

using namespace mlir;
using namespace heir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<arith_ext::ArithExtDialect>();
  registry.insert<bgv::BGVDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<cggi::CGGIDialect>();
  registry.insert<poly_ext::PolyExtDialect>();
  registry.insert<secret::SecretDialect>();
  registry.insert<tfhe_rust::TfheRustDialect>();
  registry.insert<tfhe_rust_bool::TfheRustBoolDialect>();
  registry.insert<openfhe::OpenfheDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();

  // Add expected MLIR dialects to the registry.
  registerAllDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
