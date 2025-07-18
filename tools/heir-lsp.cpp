#include "lib/Dialect/BGV/IR/BGVDialect.h"
#include "lib/Dialect/CGGI/IR/CGGIDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/Jaxite/IR/JaxiteDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/MathExt/IR/MathExtDialect.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/Random/IR/RandomDialect.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"
#include "lib/Dialect/TfheRustBool/IR/TfheRustBoolDialect.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Bufferization/IR/Bufferization.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tosa/IR/TosaOps.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project

using namespace mlir;
using namespace heir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<bgv::BGVDialect>();
  registry.insert<ckks::CKKSDialect>();
  registry.insert<cggi::CGGIDialect>();
  registry.insert<comb::CombDialect>();
  registry.insert<jaxite::JaxiteDialect>();
  registry.insert<jaxiteword::JaxiteWordDialect>();
  registry.insert<lattigo::LattigoDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<mod_arith::ModArithDialect>();
  registry.insert<mgmt::MgmtDialect>();
  registry.insert<random::RandomDialect>();
  registry.insert<openfhe::OpenfheDialect>();
  registry.insert<rns::RNSDialect>();
  registry.insert<secret::SecretDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();
  registry.insert<tfhe_rust::TfheRustDialect>();
  registry.insert<tfhe_rust_bool::TfheRustBoolDialect>();
  registry.insert<math_ext::MathExtDialect>();

  rns::registerExternalRNSTypeInterfaces(registry);

  // Add expected MLIR dialects to the registry.
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<::mlir::linalg::LinalgDialect>();
  registry.insert<::mlir::tosa::TosaDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<polynomial::PolynomialDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
