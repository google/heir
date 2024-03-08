#include "include/Dialect/BGV/IR/BGVDialect.h"
#include "include/Dialect/CGGI/IR/CGGIDialect.h"
#include "include/Dialect/Comb/IR/CombDialect.h"
#include "include/Dialect/LWE/IR/LWEDialect.h"
#include "include/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "include/Dialect/PolyExt/IR/PolyExtDialect.h"
#include "include/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "include/Dialect/Secret/IR/SecretDialect.h"
#include "include/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "include/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "include/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"     // from @llvm-project
#include "mlir/include/mlir/InitAllDialects.h"             // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project

using namespace mlir;
using namespace heir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<bgv::BGVDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<cggi::CGGIDialect>();
  registry.insert<poly_ext::PolyExtDialect>();
  registry.insert<polynomial::PolynomialDialect>();
  registry.insert<secret::SecretDialect>();
  registry.insert<tfhe_rust::TfheRustDialect>();
  registry.insert<tfhe_rust_bool::TfheRustBoolDialect>();
  registry.insert<openfhe::OpenfheDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();

  // Add expected MLIR dialects to the registry.
  registerAllDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
