#ifndef LIB_TARGET_OPENFHEBIN_OPENFHEBINEMITTER_H_
#define LIB_TARGET_OPENFHEBIN_OPENFHEBINEMITTER_H_

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir::heir::openfhe {

void registerToOpenFheBinTranslation();
LogicalResult translateToOpenFheBin(mlir::Operation *op, llvm::raw_ostream &os);

class OpenFheBinEmitter {
 public:
  OpenFheBinEmitter(raw_ostream &os, SelectVariableNames *variableNames)
      : os(os), variableNames(variableNames) {}
  LogicalResult translate(::mlir::Operation &operation);

 private:
  LogicalResult printOperation(mlir::ModuleOp module);
  LogicalResult printOperation(memref::LoadOp load);
  LogicalResult printOperation(memref::StoreOp store);
  LogicalResult printOperation(memref::AllocOp alloc);
  LogicalResult printOperation(memref::SubViewOp subview);
  LogicalResult printOperation(memref::CopyOp copy);
  LogicalResult printOperation(openfhe::GetLWESchemeOp getScheme);
  LogicalResult printOperation(openfhe::LWEMulConstOp mul);
  LogicalResult printOperation(openfhe::LWEAddOp add);
  LogicalResult printOperation(openfhe::MakeLutOp makeLut);
  LogicalResult printOperation(openfhe::EvalFuncOp evalFunc);

  LogicalResult printOperation(lwe::EncodeOp encode);
  LogicalResult printOperation(lwe::TrivialEncryptOp trivialEncrypt);

  LogicalResult printOperation(comb::InvOp op);

  // Function operations
  LogicalResult printOperation(func::FuncOp funcOp);
  LogicalResult printOperation(func::CallOp callOp);
  LogicalResult printOperation(func::ReturnOp returnOp);

  // Arith operations
  LogicalResult printOperation(arith::ConstantOp constantOp);

  // Tensor operations
  LogicalResult printOperation(tensor::ExtractOp extractOp);
  LogicalResult printOperation(tensor::FromElementsOp fromElementsOp);

  // some of the control flow ops
  LogicalResult printOperation(scf::IfOp ifOp);
  LogicalResult printOperation(affine::AffineForOp forOp);

  LogicalResult printOperation(memref::ReinterpretCastOp castOp);
  LogicalResult printOperation(memref::CollapseShapeOp collapseOp);

  LogicalResult printInPlaceEvalMethod(mlir::Value result,
                                       mlir::Value cryptoContext,
                                       mlir::ValueRange operands,
                                       std::string_view op, Location loc);

  // Helper methods for code generation
  void emitAutoAssignPrefix(::mlir::Value result);
  LogicalResult emitTypedAssignPrefix(::mlir::Value result,
                                      ::mlir::Location loc,
                                      bool constant = true);
  LogicalResult emitType(::mlir::Type type, ::mlir::Location loc,
                         bool constant = true);
  LogicalResult printEvalMethod(::mlir::Value result,
                                ::mlir::Value cryptoContext,
                                ::mlir::ValueRange nonEvalOperands,
                                std::string_view op);

  /// Output stream to emit to.
  raw_ostream &os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames *variableNames;

  /// Set of values that are mutable and don't need assign prefixes.
  llvm::DenseSet<::mlir::Value> mutableValues;

  SmallVector<std::string> getStaticDynamicArgs(
      SmallVector<mlir::Value> dynamicArgs, ArrayRef<int64_t> staticArgs);

  template <class T, typename = std::enable_if_t<std::disjunction<
                         std::is_same<T, memref::SubViewOp>,
                         std::is_same<T, memref::ReinterpretCastOp>>::value>>
  std::string getSubviewArgs(T op);

  mlir::FailureOr<std::string> getAllocConstructor(MemRefType type,
                                                   Location loc);

  // Fallback translation for operations not handled by this emitter
  LogicalResult translateFallback(::mlir::Operation &operation);
};

}  // namespace mlir::heir::openfhe

#endif  // LIB_TARGET_OPENFHEBIN_OPENFHEBINEMITTER_H_
