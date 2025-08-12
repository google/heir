#include "lib/Dialect/TfheRust/IR/TfheRustDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"              // from @llvm-project

// Force import order
#include "lib/Dialect/TfheRust/IR/TfheRustDialect.cpp.inc"
#include "lib/Dialect/TfheRust/IR/TfheRustOps.h"
#include "lib/Dialect/TfheRust/IR/TfheRustPatterns.h"
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.h"
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/TfheRust/IR/TfheRustOps.cpp.inc"

namespace mlir {
namespace heir {
namespace tfhe_rust {

void TfheRustDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/TfheRust/IR/TfheRustTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/TfheRust/IR/TfheRustOps.cpp.inc"
      >();
}

void GenerateLookupTableOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.add<HoistGenerateLookupTable>(context);
}

void CreateTrivialOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                  MLIRContext* context) {
  results.add<HoistCreateTrivial>(context);
}

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir
