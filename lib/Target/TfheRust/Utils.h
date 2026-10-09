#ifndef LIB_TARGET_TFHERUST_UTILS_H_
#define LIB_TARGET_TFHERUST_UTILS_H_

#include <cstdint>
#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace tfhe_rust {

// Determine if the func can be emitted for tfhe-rs. If not, emit a
// warning and return success. This is because some functions are left
// over during compilation.
::mlir::LogicalResult canEmitFuncForTfheRust(::mlir::func::FuncOp& funcOp);
int16_t getTfheRustBitWidth(Type type);

// getRustIntegerType returns the width of the closest builtin integer type.
FailureOr<int> getRustIntegerType(int width);

// getRustIntegerTypeStr returns the Rust integer type string (e.g. "bool",
// "i8", "u16").
FailureOr<std::string> getRustIntegerTypeStr(IntegerType type);

// getConstantGlobalData retrieves constant global data from a GetGlobalOp if
// available.
FailureOr<DenseElementsAttr> getConstantGlobalData(memref::GetGlobalOp op);

// Shared printer for memref::GetGlobalOp.
LogicalResult printGetGlobalOp(
    memref::GetGlobalOp op, llvm::raw_ostream& os,
    SelectVariableNames* variableNames,
    llvm::function_ref<FailureOr<std::string>(Type)> convertType);

// Shared printer for arith::ConstantOp (used by TfheRust and TfheRustBool
// emitters).
LogicalResult printConstantOp(arith::ConstantOp op, llvm::raw_ostream& os,
                              SelectVariableNames* variableNames);

// Shared printer for arith::IndexCastOp.
LogicalResult printIndexCastOp(
    arith::IndexCastOp op, llvm::raw_ostream& os,
    SelectVariableNames* variableNames,
    llvm::function_ref<LogicalResult(Type)> emitType);

// Shared printer for arith::TruncIOp.
LogicalResult printTruncIOp(arith::TruncIOp op, llvm::raw_ostream& os,
                            SelectVariableNames* variableNames,
                            llvm::function_ref<LogicalResult(Type)> emitType);

}  // namespace tfhe_rust
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_TFHERUST_UTILS_H_
