#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "include/cereal/cereal.hpp"        // from @cereal
#include "include/cereal/types/map.hpp"     // from @cereal
#include "include/cereal/types/string.hpp"  // from @cereal
#include "include/cereal/types/vector.hpp"  // from @cereal
#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

/// Translates the given operation to OpenFhePke.
::mlir::LogicalResult translateToOpenFhePke(::mlir::Operation *op,
                                            llvm::raw_ostream &os,
                                            const OpenfheImportType &importType,
                                            const std::string &weightsFile);

// A map from the SSA value name of a 1-D dense element constants to its value.
// Note that multidimensional shapes are handled as flattened 1-D vectors.
struct Weights {
  std::map<std::string, std::vector<float>> floats;
  std::map<std::string, std::vector<double>> doubles;
  std::map<std::string, std::vector<int64_t>> int64_ts;
  std::map<std::string, std::vector<int32_t>> int32_ts;
  std::map<std::string, std::vector<int16_t>> int16_ts;
  std::map<std::string, std::vector<int8_t>> int8_ts;

  template <class Archive>
  void serialize(Archive &archive) {
    archive(CEREAL_NVP(floats), CEREAL_NVP(doubles), CEREAL_NVP(int64_ts),
            CEREAL_NVP(int32_ts), CEREAL_NVP(int16_ts), CEREAL_NVP(int8_ts));
  }
};

class OpenFhePkeEmitter {
 public:
  OpenFhePkeEmitter(raw_ostream &os, SelectVariableNames *variableNames,
                    const OpenfheImportType &importType,
                    const std::string &weightsFile);

  LogicalResult translate(::mlir::Operation &operation);

 private:
  OpenfheImportType importType_;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Pre-populated analysis selecting unique variable names for all the SSA
  /// values.
  SelectVariableNames *variableNames;

  /// Set of values that are mutable and don't need assign prefixes.
  llvm::DenseSet<::mlir::Value> mutableValues;

  // Module containing global weights
  Weights weightsMap_;

  const std::string &weightsFile_;

  // Functions for printing individual ops
  LogicalResult printOperation(::mlir::ModuleOp op);
  LogicalResult printOperation(::mlir::affine::AffineForOp op);
  LogicalResult printOperation(::mlir::affine::AffineYieldOp op);
  LogicalResult printOperation(::mlir::arith::AddIOp op);
  LogicalResult printOperation(::mlir::arith::CmpIOp op);
  LogicalResult printOperation(::mlir::arith::ConstantOp op);
  LogicalResult printOperation(::mlir::arith::ExtFOp op);
  LogicalResult printOperation(::mlir::arith::ExtSIOp op);
  LogicalResult printOperation(::mlir::arith::ExtUIOp op);
  LogicalResult printOperation(::mlir::arith::IndexCastOp op);
  LogicalResult printOperation(::mlir::arith::RemSIOp op);
  LogicalResult printOperation(::mlir::arith::SelectOp op);
  LogicalResult printOperation(::mlir::tensor::EmptyOp op);
  LogicalResult printOperation(::mlir::tensor::ExtractOp op);
  LogicalResult printOperation(::mlir::tensor::ExtractSliceOp op);
  LogicalResult printOperation(::mlir::tensor::InsertOp op);
  LogicalResult printOperation(::mlir::tensor::SplatOp op);
  LogicalResult printOperation(::mlir::func::FuncOp op);
  LogicalResult printOperation(::mlir::func::CallOp op);
  LogicalResult printOperation(::mlir::func::ReturnOp op);
  LogicalResult printOperation(::mlir::heir::lwe::RLWEDecodeOp op);
  LogicalResult printOperation(
      ::mlir::heir::lwe::ReinterpretApplicationDataOp op);
  LogicalResult printOperation(AddOp op);
  LogicalResult printOperation(AddPlainOp op);
  LogicalResult printOperation(AutomorphOp op);
  LogicalResult printOperation(BootstrapOp op);
  LogicalResult printOperation(DecryptOp op);
  LogicalResult printOperation(EncryptOp op);
  LogicalResult printOperation(GenParamsOp op);
  LogicalResult printOperation(GenContextOp op);
  LogicalResult printOperation(GenMulKeyOp op);
  LogicalResult printOperation(GenRotKeyOp op);
  LogicalResult printOperation(GenBootstrapKeyOp op);
  LogicalResult printOperation(KeySwitchOp op);
  LogicalResult printOperation(LevelReduceOp op);
  LogicalResult printOperation(MakePackedPlaintextOp op);
  LogicalResult printOperation(MakeCKKSPackedPlaintextOp op);
  LogicalResult printOperation(ModReduceOp op);
  LogicalResult printOperation(MulConstOp op);
  LogicalResult printOperation(MulNoRelinOp op);
  LogicalResult printOperation(MulOp op);
  LogicalResult printOperation(MulPlainOp op);
  LogicalResult printOperation(NegateOp op);
  LogicalResult printOperation(RelinOp op);
  LogicalResult printOperation(RotOp op);
  LogicalResult printOperation(SetupBootstrapOp op);
  LogicalResult printOperation(SquareOp op);
  LogicalResult printOperation(SubOp op);
  LogicalResult printOperation(SubPlainOp op);

  // Helpers for above
  LogicalResult printEvalMethod(::mlir::Value result,
                                ::mlir::Value cryptoContext,
                                ::mlir::ValueRange nonEvalOperands,
                                std::string_view op);
  LogicalResult printBinaryOp(Operation *op, ::mlir::Value lhs,
                              ::mlir::Value rhs, std::string_view opName);

  // Emit an OpenFhe type, using a const specifier.
  LogicalResult emitType(::mlir::Type type, ::mlir::Location loc,
                         bool constant = true);

  // Canonicalize Debug Port
  bool isDebugPort(::llvm::StringRef debugPortName);
  ::llvm::StringRef canonicalizeDebugPort(::llvm::StringRef debugPortName);

  std::string getDebugAttrMapName() {
    static int debugAttrMapCount = 0;
    return "debugAttrMap" + std::to_string(debugAttrMapCount++);
  }

  void emitAutoAssignPrefix(::mlir::Value result);
  LogicalResult emitTypedAssignPrefix(::mlir::Value result,
                                      ::mlir::Location loc,
                                      bool constant = true);
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKEEMITTER_H_
