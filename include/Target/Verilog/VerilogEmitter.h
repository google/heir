#ifndef THIRD_PARTY_HEIR_INCLUDE_TARGET_VERILOG_VERILOGEMITTER_H_
#define THIRD_PARTY_HEIR_INCLUDE_TARGET_VERILOG_VERILOGEMITTER_H_

#include "llvm/include/llvm/ADT/DenseMap.h" // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h" // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h" // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h" // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h" // from @llvm-project

namespace mlir {
namespace heir {

void registerToVerilogTranslation();

/// Translates the given operation to Verilog.
mlir::LogicalResult translateToVerilog(mlir::Operation *op,
                                       llvm::raw_ostream &os);

class VerilogEmitter {
 public:
  VerilogEmitter(raw_ostream &os);

  LogicalResult translate(mlir::Operation &operation);

 private:
  /// Output stream to emit to.
  raw_indented_ostream os_;

  /// Map from a Value to name of Verilog variable that is bound to the value.
  llvm::DenseMap<Value, std::string> value_to_wire_name_;

  // Globally unique identifiers for values
  int64_t value_count_;

  // Functions for printing individual ops
  LogicalResult printOperation(mlir::ModuleOp op);
  LogicalResult printOperation(mlir::UnrealizedConversionCastOp op);
  LogicalResult printOperation(mlir::arith::AddIOp op);
  LogicalResult printOperation(mlir::arith::AndIOp op);
  LogicalResult printOperation(mlir::arith::CmpIOp op);
  LogicalResult printOperation(mlir::arith::ConstantOp op);
  LogicalResult printOperation(mlir::arith::ExtSIOp op);
  LogicalResult printOperation(mlir::arith::ExtUIOp op);
  LogicalResult printOperation(mlir::arith::IndexCastOp op);
  LogicalResult printOperation(mlir::arith::MulIOp op);
  LogicalResult printOperation(mlir::arith::SelectOp op);
  LogicalResult printOperation(mlir::arith::ShLIOp op);
  LogicalResult printOperation(mlir::arith::ShRSIOp op);
  LogicalResult printOperation(mlir::arith::ShRUIOp op);
  LogicalResult printOperation(mlir::arith::SubIOp op);
  LogicalResult printOperation(mlir::arith::TruncIOp op);
  LogicalResult printOperation(mlir::affine::AffineLoadOp op);
  LogicalResult printOperation(mlir::affine::AffineStoreOp op);
  LogicalResult printOperation(mlir::func::CallOp op);
  LogicalResult printOperation(mlir::func::FuncOp op);
  LogicalResult printOperation(mlir::func::ReturnOp op);
  LogicalResult printOperation(mlir::math::CountLeadingZerosOp op);
  LogicalResult printOperation(mlir::memref::LoadOp op);

  // Helpers for above
  LogicalResult printBinaryOp(mlir::Value result, mlir::Value lhs,
                              mlir::Value rhs, std::string_view op);

  // Emit a Verilog type of the form `wire [width-1:0]`
  LogicalResult emitType(Location loc, Type type);

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
  StringRef getName(Value value);
};

}  // namespace heir
}  // namespace mlir

#endif  // THIRD_PARTY_HEIR_INCLUDE_TARGET_VERILOG_VERILOGEMITTER_H_
