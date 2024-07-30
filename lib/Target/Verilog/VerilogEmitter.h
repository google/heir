#ifndef LIB_TARGET_VERILOG_VERILOGEMITTER_H_
#define LIB_TARGET_VERILOG_VERILOGEMITTER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/ilist.h"            // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/TypeRange.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {

void registerToVerilogTranslation();

/// Translates the given operation to Verilog.
mlir::LogicalResult translateToVerilog(mlir::Operation *op,
                                       llvm::raw_ostream &os);

/// Translates the given operation to Verilog with a fixed input name for the
/// resulting verilog module. Raises an error if the input IR contains secret
/// ops.
mlir::LogicalResult translateToVerilog(
    mlir::Operation *op, llvm::raw_ostream &os,
    std::optional<llvm::StringRef> moduleName);

/// Translates the given operation to Verilog with a fixed input name for the
/// resulting verilog module. If allowSecretOps is false, raises an error if
/// the input IR contains secret ops.
mlir::LogicalResult translateToVerilog(
    mlir::Operation *op, llvm::raw_ostream &os,
    std::optional<llvm::StringRef> moduleName, bool allowSecretOps);

class VerilogEmitter {
 public:
  VerilogEmitter(raw_ostream &os);

  LogicalResult translate(mlir::Operation &operation,
                          std::optional<llvm::StringRef> moduleName);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os_;

  /// Map from a Value to name of Verilog variable that is bound to the value.
  llvm::DenseMap<Value, std::string> value_to_wire_name_;
  llvm::SmallVector<std::string> output_wire_names_;

  // Globally unique identifiers for values
  int64_t value_count_;

  // A helper to generalize the work of emitting a func.func and a
  // secret.generic
  LogicalResult printFunctionLikeOp(Operation *op,
                                    llvm::StringRef verilogModuleName,
                                    ArrayRef<BlockArgument> arguments,
                                    TypeRange resultTypes,
                                    Region::BlockListType::iterator blocksBegin,
                                    Region::BlockListType::iterator blocksEnd);

  // A helper to generalize the work of emitting a func.return and a
  // secret.yield
  LogicalResult printReturnLikeOp(ValueRange returnValues);

  // Functions for printing individual ops
  LogicalResult printOperation(mlir::ModuleOp op,
                               std::optional<llvm::StringRef> moduleName);
  LogicalResult printOperation(mlir::func::FuncOp op,
                               std::optional<llvm::StringRef> moduleName);
  LogicalResult printOperation(mlir::heir::secret::GenericOp op,
                               std::optional<llvm::StringRef> moduleName);
  LogicalResult printOperation(mlir::UnrealizedConversionCastOp op);
  LogicalResult printOperation(mlir::arith::AddIOp op);
  LogicalResult printOperation(mlir::arith::AndIOp op);
  LogicalResult printOperation(mlir::arith::CmpIOp op);
  LogicalResult printOperation(mlir::arith::ConstantOp op);
  LogicalResult printOperation(mlir::arith::ExtSIOp op);
  LogicalResult printOperation(mlir::arith::ExtUIOp op);
  LogicalResult printOperation(mlir::arith::IndexCastOp op);
  LogicalResult printOperation(mlir::arith::MaxSIOp op);
  LogicalResult printOperation(mlir::arith::MinSIOp op);
  LogicalResult printOperation(mlir::arith::MulIOp op);
  LogicalResult printOperation(mlir::arith::SelectOp op);
  LogicalResult printOperation(mlir::arith::ShLIOp op);
  LogicalResult printOperation(mlir::arith::ShRSIOp op);
  LogicalResult printOperation(mlir::arith::ShRUIOp op);
  LogicalResult printOperation(mlir::arith::SubIOp op);
  LogicalResult printOperation(mlir::arith::TruncIOp op);
  LogicalResult printOperation(mlir::affine::AffineLoadOp op);
  LogicalResult printOperation(mlir::affine::AffineParallelOp op);
  LogicalResult printOperation(mlir::affine::AffineStoreOp op);
  LogicalResult printOperation(mlir::affine::AffineYieldOp op);
  LogicalResult printOperation(mlir::func::CallOp op);
  LogicalResult printOperation(mlir::math::CountLeadingZerosOp op);
  LogicalResult printOperation(mlir::memref::StoreOp op);
  LogicalResult printOperation(mlir::memref::LoadOp op);

  // Helpers for above
  LogicalResult printBinaryOp(mlir::Value result, mlir::Value lhs,
                              mlir::Value rhs, std::string_view op);

  // Emit a Verilog type of the form `wire [width-1:0]`
  LogicalResult emitType(Type type);
  LogicalResult emitType(Type type, raw_ostream &os);
  LogicalResult emitIndexType(Value indexValue, raw_ostream &os);

  // Emit a Verilog array shape specifier of the form `[width]`
  LogicalResult emitArrayShapeSuffix(Type type);

  // Emit a wire declaration in the verilog module body
  LogicalResult emitWireDeclaration(OpResult result);

  // Emit `assign var_name = ` as a common prefix to
  // many printOperation functions above.
  void emitAssignPrefix(mlir::Value result);

  // Get or create a variable name
  StringRef getOrCreateName(BlockArgument arg);
  StringRef getOrCreateName(Value value);
  StringRef getOrCreateName(Value value, std::string_view prefix);
  StringRef getOrCreateOutputWireName(int resultIndex);
  StringRef getOutputWireName(int resultIndex);
  StringRef getName(Value value);
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_VERILOG_VERILOGEMITTER_H_
