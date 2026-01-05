#include "lib/Target/OpenFhePke/Interpreter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"       // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"      // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"       // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"            // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Parser/Parser.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "src/core/include/lattice/hal/lat-backend.h"    // from @openfhe
#include "src/core/include/lattice/stdlatticeparms.h"    // from @openfhe
#include "src/pke/include/ciphertext-fwd.h"              // from @openfhe
#include "src/pke/include/constants-defs.h"              // from @openfhe
#include "src/pke/include/cryptocontext-fwd.h"           // from @openfhe
#include "src/pke/include/encoding/plaintext-fwd.h"      // from @openfhe
#include "src/pke/include/gen-cryptocontext.h"           // from @openfhe
#include "src/pke/include/key/evalkey-fwd.h"             // from @openfhe
#include "src/pke/include/key/privatekey-fwd.h"          // from @openfhe
#include "src/pke/include/key/publickey-fwd.h"           // from @openfhe
#include "src/pke/include/openfhe.h"                     // from @openfhe

#ifdef OPENFHE_ENABLE_TIMING
#define TIME_OPERATION_VOID(op_name, code)                  \
  do {                                                      \
    auto start = std::chrono::high_resolution_clock::now(); \
    code;                                                   \
    auto end = std::chrono::high_resolution_clock::now();   \
    timingResults[op_name].totalTime += (end - start);      \
    timingResults[op_name].count++;                         \
  } while (0)
#define TIME_OPERATION(op_name, result_val, code)                \
  do {                                                           \
    auto start = std::chrono::high_resolution_clock::now();      \
    auto result = code;                                          \
    auto end = std::chrono::high_resolution_clock::now();        \
    timingResults[op_name].totalTime += (end - start);           \
    timingResults[op_name].count++;                              \
    ciphertexts.insert_or_assign(result_val, std::move(result)); \
  } while (0)
#define TIME_OPERATION_NONCT(op_name, result_val, code, map) \
  do {                                                       \
    auto start = std::chrono::high_resolution_clock::now();  \
    auto result = code;                                      \
    auto end = std::chrono::high_resolution_clock::now();    \
    timingResults[op_name].totalTime += (end - start);       \
    timingResults[op_name].count++;                          \
    (map).insert_or_assign(result_val, std::move(result));   \
  } while (0)
#else
#define TIME_OPERATION_VOID(op_name, code) \
  do {                                     \
    code;                                  \
  } while (0)
#define TIME_OPERATION(op_name, result_val, code)   \
  do {                                              \
    ciphertexts.insert_or_assign(result_val, code); \
  } while (0)
#define TIME_OPERATION_NONCT(op_name, result_val, code, map) \
  do {                                                       \
    (map).insert_or_assign(result_val, code);                \
  } while (0)
#endif

namespace mlir {
namespace heir {
namespace openfhe {

using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;
using FastRotPrecompT = std::shared_ptr<std::vector<DCRTPoly>>;

// Helper function for floor division on integers
static inline int floorDivInt(int lhs, int rhs) {
  return static_cast<int>(std::floor(static_cast<float>(lhs) / rhs));
}

// Static member initialization
llvm::DenseMap<TypeID, Interpreter::OperationVisitor>
    Interpreter::operationDispatchTable;
bool Interpreter::dispatchTableInitialized = false;
MLIRContext* Interpreter::dispatchTableContext = nullptr;

void Interpreter::initializeDispatchTable() {
  auto* ctx = module.getContext();

  // Reinitialize if context changed or not yet initialized
  if (dispatchTableInitialized && dispatchTableContext == ctx) {
    return;
  }

  // Clear the dispatch table if context changed
  if (dispatchTableContext != ctx) {
    operationDispatchTable.clear();
  }

  dispatchTableContext = ctx;

// Helper macro to register an operation type
#define REGISTER_OP(OpType)                                               \
  operationDispatchTable[TypeID::get<OpType>()] = [](Interpreter* interp, \
                                                     Operation* op) {     \
    interp->visit(llvm::cast<OpType>(op));                                \
  };

  // Register all supported operations
  REGISTER_OP(arith::ConstantOp);
  REGISTER_OP(arith::AddIOp);
  REGISTER_OP(arith::AddFOp);
  REGISTER_OP(arith::SubIOp);
  REGISTER_OP(arith::SubFOp);
  REGISTER_OP(arith::MulIOp);
  REGISTER_OP(arith::MulFOp);
  REGISTER_OP(arith::DivSIOp);
  REGISTER_OP(arith::FloorDivSIOp);
  REGISTER_OP(arith::RemSIOp);
  REGISTER_OP(arith::AndIOp);
  REGISTER_OP(arith::CmpIOp);
  REGISTER_OP(arith::SelectOp);
  REGISTER_OP(arith::ExtFOp);
  REGISTER_OP(arith::MinSIOp);
  REGISTER_OP(arith::MaxSIOp);
  REGISTER_OP(tensor::EmptyOp);
  REGISTER_OP(tensor::ExtractOp);
  REGISTER_OP(tensor::InsertOp);
  REGISTER_OP(tensor::SplatOp);
  REGISTER_OP(tensor::FromElementsOp);
  REGISTER_OP(tensor::ConcatOp);
  REGISTER_OP(tensor::ExtractSliceOp);
  REGISTER_OP(tensor::InsertSliceOp);
  REGISTER_OP(tensor::CollapseShapeOp);
  REGISTER_OP(tensor::ExpandShapeOp);
  REGISTER_OP(linalg::BroadcastOp);
  REGISTER_OP(scf::ForOp);
  REGISTER_OP(scf::IfOp);
  REGISTER_OP(scf::YieldOp);
  REGISTER_OP(affine::AffineForOp);
  REGISTER_OP(affine::AffineYieldOp);
  REGISTER_OP(lwe::RLWEDecodeOp);
  REGISTER_OP(AddOp);
  REGISTER_OP(AddPlainOp);
  REGISTER_OP(SubOp);
  REGISTER_OP(SubPlainOp);
  REGISTER_OP(MulOp);
  REGISTER_OP(MulNoRelinOp);
  REGISTER_OP(MulPlainOp);
  REGISTER_OP(MulConstOp);
  REGISTER_OP(NegateOp);
  REGISTER_OP(SquareOp);
  REGISTER_OP(RelinOp);
  REGISTER_OP(ModReduceOp);
  REGISTER_OP(LevelReduceOp);
  REGISTER_OP(RotOp);
  REGISTER_OP(AutomorphOp);
  REGISTER_OP(KeySwitchOp);
  REGISTER_OP(BootstrapOp);
  REGISTER_OP(EncryptOp);
  REGISTER_OP(DecryptOp);
  REGISTER_OP(MakePackedPlaintextOp);
  REGISTER_OP(MakeCKKSPackedPlaintextOp);
  REGISTER_OP(GenParamsOp);
  REGISTER_OP(GenContextOp);
  REGISTER_OP(GenRotKeyOp);
  REGISTER_OP(GenMulKeyOp);
  REGISTER_OP(GenBootstrapKeyOp);
  REGISTER_OP(SetupBootstrapOp);
  REGISTER_OP(FastRotationOp);
  REGISTER_OP(FastRotationPrecomputeOp);

#undef REGISTER_OP

  dispatchTableInitialized = true;
}

void Interpreter::eraseValue(Value v) {
  llvm::TypeSwitch<Type>(v.getType())
      .Case<lwe::LWEPlaintextType>([&](auto ty) { plaintexts.erase(v); })
      .Case<lwe::LWECiphertextType>([&](auto ty) { ciphertexts.erase(v); })
      .Case<RankedTensorType>([&](auto ty) {
        auto elemType = ty.getElementType();
        if (elemType.isInteger() || elemType.isIndex()) {
          intVectors.erase(v);
        } else if (elemType.isF32()) {
          floatVectors.erase(v);
        } else if (elemType.isF64()) {
          doubleVectors.erase(v);
        } else if (isa<lwe::LWEPlaintextType>(elemType)) {
          plaintextVectors.erase(v);
        } else if (isa<lwe::LWECiphertextType>(elemType)) {
          ciphertextVectors.erase(v);
        } else {
          llvm::errs() << "Unsupported tensor element type " << elemType
                       << " in eraseValue\n";
        }
      })
      .Case<IntegerType, IndexType>([&](auto ty) {
        if (ty.isInteger(1)) {
          boolValues.erase(v);
        } else {
          intValues.erase(v);
        }
      })
      .Case<FloatType>([&](auto ty) {
        if (ty.isF32()) {
          floatValues.erase(v);
        } else {
          doubleValues.erase(v);
        }
      })
      .Case<openfhe::DigitDecompositionType>(
          [&](auto ty) { fastRotPrecomps.erase(v); })
      .Case<openfhe::EvalKeyType>([&](auto ty) { evalKeys.erase(v); })
      .Case<openfhe::PublicKeyType>([&](auto ty) { publicKeys.erase(v); })
      .Case<openfhe::PrivateKeyType>([&](auto ty) { privateKeys.erase(v); })
      .Case<openfhe::CryptoContextType>(
          [&](auto ty) { cryptoContexts.erase(v); })
      .Default([&](Type type) {
        llvm::errs() << "Unsupported type " << type << " in eraseValue\n";
      });
}

// Only used for type-agnostic block arguments (func args, iter args, etc.)
void Interpreter::storeTypedValue(Value v, const TypedCppValue& typedVal) {
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          // Nothing to store
        } else if constexpr (std::is_same_v<T, bool>) {
          boolValues[v] = arg;
        } else if constexpr (std::is_same_v<T, int>) {
          intValues[v] = arg;
        } else if constexpr (std::is_same_v<T, float>) {
          floatValues[v] = arg;
        } else if constexpr (std::is_same_v<T, double>) {
          doubleValues[v] = arg;
        } else if constexpr (std::is_same_v<
                                 T, std::shared_ptr<std::vector<int>>>) {
          intVectors[v] = arg;
        } else if constexpr (std::is_same_v<
                                 T, std::shared_ptr<std::vector<float>>>) {
          floatVectors[v] = arg;
        } else if constexpr (std::is_same_v<
                                 T, std::shared_ptr<std::vector<double>>>) {
          doubleVectors[v] = arg;
        } else if constexpr (std::is_same_v<T, PlaintextT>) {
          plaintexts[v] = arg;
        } else if constexpr (std::is_same_v<
                                 T, std::shared_ptr<std::vector<PlaintextT>>>) {
          plaintextVectors[v] = arg;
        } else if constexpr (std::is_same_v<T, CiphertextT>) {
          ciphertexts[v] = arg;
        } else if constexpr (std::is_same_v<T, std::shared_ptr<
                                                   std::vector<CiphertextT>>>) {
          ciphertextVectors[v] = arg;
        } else if constexpr (std::is_same_v<T, PublicKeyT>) {
          publicKeys[v] = arg;
        } else if constexpr (std::is_same_v<T, PrivateKeyT>) {
          privateKeys[v] = arg;
        } else if constexpr (std::is_same_v<T, EvalKeyT>) {
          evalKeys[v] = arg;
        } else if constexpr (std::is_same_v<T, CryptoContextT>) {
          cryptoContexts[v] = arg;
        } else if constexpr (std::is_same_v<T, FastRotPrecompT>) {
          fastRotPrecomps[v] = arg;
        }
      },
      typedVal.value);
}

// Only used for type-agnostic block terminators (return, yield)
TypedCppValue Interpreter::loadTypedValue(Value v) {
  TypedCppValue result;

  llvm::TypeSwitch<Type>(v.getType())
      .Case<lwe::LWEPlaintextType>([&](auto ty) {
        if (auto it = plaintexts.find(v); it != plaintexts.end())
          result = TypedCppValue(it->second);
      })
      .Case<lwe::LWECiphertextType>([&](auto ty) {
        if (auto it = ciphertexts.find(v); it != ciphertexts.end())
          result = TypedCppValue(it->second);
      })
      .Case<RankedTensorType>([&](auto ty) {
        auto elemType = ty.getElementType();
        if (elemType.isInteger() || elemType.isIndex()) {
          if (auto it = intVectors.find(v); it != intVectors.end())
            result = TypedCppValue(it->second);
        } else if (elemType.isF32()) {
          if (auto it = floatVectors.find(v); it != floatVectors.end())
            result = TypedCppValue(it->second);
        } else if (elemType.isF64()) {
          if (auto it = doubleVectors.find(v); it != doubleVectors.end())
            result = TypedCppValue(it->second);
        } else if (isa<lwe::LWEPlaintextType>(elemType)) {
          if (auto it = plaintextVectors.find(v); it != plaintextVectors.end())
            result = TypedCppValue(it->second);
        } else if (isa<lwe::LWECiphertextType>(elemType)) {
          if (auto it = ciphertextVectors.find(v);
              it != ciphertextVectors.end())
            result = TypedCppValue(it->second);
        }
      })
      .Case<IntegerType, IndexType>([&](auto ty) {
        if (ty.isInteger(1)) {
          if (auto it = boolValues.find(v); it != boolValues.end())
            result = TypedCppValue(it->second);
        } else {
          if (auto it = intValues.find(v); it != intValues.end())
            result = TypedCppValue(it->second);
        }
      })
      .Case<FloatType>([&](auto ty) {
        if (ty.isF32()) {
          if (auto it = floatValues.find(v); it != floatValues.end())
            result = TypedCppValue(it->second);
        } else {
          if (auto it = doubleValues.find(v); it != doubleValues.end())
            result = TypedCppValue(it->second);
        }
      })
      .Case<openfhe::DigitDecompositionType>([&](auto ty) {
        if (auto it = fastRotPrecomps.find(v); it != fastRotPrecomps.end())
          result = TypedCppValue(it->second);
      })
      .Case<openfhe::EvalKeyType>([&](auto ty) {
        if (auto it = evalKeys.find(v); it != evalKeys.end())
          result = TypedCppValue(it->second);
      })
      .Case<openfhe::PublicKeyType>([&](auto ty) {
        if (auto it = publicKeys.find(v); it != publicKeys.end())
          result = TypedCppValue(it->second);
      })
      .Case<openfhe::PrivateKeyType>([&](auto ty) {
        if (auto it = privateKeys.find(v); it != privateKeys.end())
          result = TypedCppValue(it->second);
      })
      .Case<openfhe::CryptoContextType>([&](auto ty) {
        if (auto it = cryptoContexts.find(v); it != cryptoContexts.end())
          result = TypedCppValue(it->second);
      })
      .Default([&](Type type) {
        llvm::errs() << "Unsupported type " << type << " in loadTypedValue\n";
      });

  return result;
}

std::vector<TypedCppValue> Interpreter::interpret(
    const std::string& entryFunction, ArrayRef<TypedCppValue> inputValues) {
  if (!dispatchTableInitialized ||
      dispatchTableContext != module.getContext()) {
    initializeDispatchTable();
  }

  // Clear all storage
  boolValues.clear();
  intValues.clear();
  floatValues.clear();
  doubleValues.clear();
  intVectors.clear();
  floatVectors.clear();
  doubleVectors.clear();
  plaintexts.clear();
  plaintextVectors.clear();
  ciphertexts.clear();
  ciphertextVectors.clear();
  cryptoContexts.clear();
  publicKeys.clear();
  privateKeys.clear();
  evalKeys.clear();
  fastRotPrecomps.clear();
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(entryFunction);
  std::vector<TypedCppValue> results;

  FunctionType funcType = func.getFunctionType();
  SmallVector<Type> argTypes(funcType.getInputs());
  SmallVector<Type> returnTypes(funcType.getResults());

  if (argTypes.size() != inputValues.size()) {
    func->emitError() << "Input size does not match function signature";
  }

  for (const auto& [argIndex, argTy] : llvm::enumerate(argTypes)) {
    storeTypedValue(func.getBody().getArgument(argIndex),
                    inputValues[argIndex]);
  }

  llvm::outs() << "Interpreting function: " << func.getName()
               << " with signature " << funcType << " and "
               << inputValues.size() << " interpreted arguments\n";

  // Walk only the operations in the entry block, not nested regions
  // Nested regions (like loop bodies) will be handled by their parent ops
  for (auto& op : func.getBody().front().getOperations()) {
    if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
      results.reserve(returnOp.getOperands().size());
      for (auto retVal : returnOp.getOperands()) {
        results.push_back(loadTypedValue(retVal));
      }
      llvm::outs() << "Function returned " << results.size() << " values\n";
    } else {
      visit(&op);
    }
  }

  return results;
}

void Interpreter::visit(Operation* op) {
  // Avoid dispatch overhead for trivial ops
  if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::IndexCastOp>(op)) {
    intValues[op->getResult(0)] = intValues.at(op->getOperand(0));
  } else {
    // Use jump table for faster dispatch
    auto it = operationDispatchTable.find(op->getName().getTypeID());
    if (it != operationDispatchTable.end()) {
      it->second(this, op);
    } else {
      op->emitError() << "Unsupported operation " << op->getName()
                      << " in interpreter\n";
    }
  }
  // If any of the operations op operands have no more uses, then remove them
  // from storage.
  if (!op->getParentOfType<affine::AffineForOp>() &&
      !op->getParentOfType<scf::ForOp>()) {
    for (auto operand : op->getOperands()) {
      if (liveness.isDeadAfter(operand, op)) {
        eraseValue(operand);
      }
    }
  }
}

void Interpreter::visit(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    intValues[op.getResult()] = static_cast<int>(intAttr.getInt());
    return;
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    // Use float for 32-bit and smaller, double for 64-bit and larger
    if (floatAttr.getType().isF32()) {
      floatValues[op.getResult()] =
          static_cast<float>(floatAttr.getValueAsDouble());
    } else {
      // F64 and larger types use double
      doubleValues[op.getResult()] = floatAttr.getValueAsDouble();
    }
    return;
  }

  DenseElementsAttr denseElementsAttr = dyn_cast<DenseElementsAttr>(valueAttr);
  if (auto denseResourceAttr = dyn_cast<DenseResourceElementsAttr>(valueAttr)) {
    const auto data = denseResourceAttr.getData();
    denseElementsAttr =
        DenseElementsAttr::getFromRawBuffer(denseResourceAttr.getType(), data);
  }

  if (denseElementsAttr) {
    if (denseElementsAttr.getType().getElementType().isF32()) {
      std::vector<float> values;
      for (auto val : denseElementsAttr.getValues<APFloat>()) {
        values.push_back(static_cast<float>(val.convertToFloat()));
      }
      floatVectors[op.getResult()] =
          std::make_shared<std::vector<float>>(values);
      return;
    }

    if (denseElementsAttr.getType().getElementType().isF64()) {
      std::vector<double> values;
      for (auto val : denseElementsAttr.getValues<APFloat>()) {
        values.push_back(val.convertToDouble());
      }
      doubleVectors[op.getResult()] =
          std::make_shared<std::vector<double>>(values);
      return;
    }

    std::vector<int> values;
    for (auto val : denseElementsAttr.getValues<APInt>()) {
      values.push_back(static_cast<int>(val.getSExtValue()));
    }
    intVectors[op.getResult()] = std::make_shared<std::vector<int>>(values);
    return;
  }

  op->emitError() << "Unsupported constant attribute type " << valueAttr
                  << "\n";
}

// Macro for handling binary operations on integer types
#define HANDLE_BINARY_OP_INT(op, binop, opname)                                \
  do {                                                                         \
    auto resultType = (op).getResult().getType();                              \
    if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {            \
      auto elemType = tensorType.getElementType();                             \
      auto numElements = tensorType.getNumElements();                          \
      if (elemType.isInteger() || elemType.isIndex()) {                        \
        const auto& lhsVec = *intVectors.at((op).getLhs());                    \
        const auto& rhsVec = *intVectors.at((op).getRhs());                    \
        auto result = std::make_shared<std::vector<int>>(numElements);         \
        for (int64_t i = 0; i < numElements; ++i) {                            \
          (*result)[i] = lhsVec[i] binop rhsVec[i];                            \
        }                                                                      \
        intVectors[(op).getResult()] = result;                                 \
      } else {                                                                 \
        (op)->emitError() << "Unsupported type for " opname ": " << resultType \
                          << "\n";                                             \
      }                                                                        \
    } else if (resultType.isInteger() || resultType.isIndex()) {               \
      intValues[(op).getResult()] =                                            \
          intValues.at((op).getLhs()) binop intValues.at((op).getRhs());       \
    } else {                                                                   \
      (op)->emitError() << "Unsupported type for " opname ": " << resultType   \
                        << "\n";                                               \
    }                                                                          \
  } while (0)

// Macro for handling binary operations on floating-point types
#define HANDLE_BINARY_OP_FLOAT(op, binop, opname)                              \
  do {                                                                         \
    auto resultType = (op).getResult().getType();                              \
    if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {            \
      auto elemType = tensorType.getElementType();                             \
      auto numElements = tensorType.getNumElements();                          \
      if (elemType.isF32()) {                                                  \
        const auto& lhsVec = *floatVectors.at((op).getLhs());                  \
        const auto& rhsVec = *floatVectors.at((op).getRhs());                  \
        auto result = std::make_shared<std::vector<float>>(numElements);       \
        for (int64_t i = 0; i < numElements; ++i) {                            \
          (*result)[i] = lhsVec[i] binop rhsVec[i];                            \
        }                                                                      \
        floatVectors[(op).getResult()] = result;                               \
      } else if (elemType.isF64()) {                                           \
        const auto& lhsVec = *doubleVectors.at((op).getLhs());                 \
        const auto& rhsVec = *doubleVectors.at((op).getRhs());                 \
        auto result = std::make_shared<std::vector<double>>(numElements);      \
        for (int64_t i = 0; i < numElements; ++i) {                            \
          (*result)[i] = lhsVec[i] binop rhsVec[i];                            \
        }                                                                      \
        doubleVectors[(op).getResult()] = result;                              \
      } else {                                                                 \
        (op)->emitError() << "Unsupported type for " opname ": " << resultType \
                          << "\n";                                             \
      }                                                                        \
    } else if (resultType.isF32()) {                                           \
      floatValues[(op).getResult()] =                                          \
          floatValues.at((op).getLhs()) binop floatValues.at((op).getRhs());   \
    } else if (resultType.isF64()) {                                           \
      doubleValues[(op).getResult()] =                                         \
          doubleValues.at((op).getLhs()) binop doubleValues.at((op).getRhs()); \
    } else {                                                                   \
      (op)->emitError() << "Unsupported type for " opname ": " << resultType   \
                        << "\n";                                               \
    }                                                                          \
  } while (0)

// Macro for handling binary operations on integer types using a function
#define HANDLE_BINARY_OP_INT_FUNC(op, func, opname)                            \
  do {                                                                         \
    auto resultType = (op).getResult().getType();                              \
    if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {            \
      auto elemType = tensorType.getElementType();                             \
      auto numElements = tensorType.getNumElements();                          \
      if (elemType.isInteger() || elemType.isIndex()) {                        \
        const auto& lhsVec = *intVectors.at((op).getLhs());                    \
        const auto& rhsVec = *intVectors.at((op).getRhs());                    \
        auto result = std::make_shared<std::vector<int>>(numElements);         \
        for (int64_t i = 0; i < numElements; ++i) {                            \
          (*result)[i] = func(lhsVec[i], rhsVec[i]);                           \
        }                                                                      \
        intVectors[(op).getResult()] = result;                                 \
      } else {                                                                 \
        (op)->emitError() << "Unsupported type for " opname ": " << resultType \
                          << "\n";                                             \
      }                                                                        \
    } else if (resultType.isInteger() || resultType.isIndex()) {               \
      intValues[(op).getResult()] =                                            \
          func(intValues.at((op).getLhs()), intValues.at((op).getRhs()));      \
    } else {                                                                   \
      (op)->emitError() << "Unsupported type for " opname ": " << resultType   \
                        << "\n";                                               \
    }                                                                          \
  } while (0)

void Interpreter::visit(arith::AddIOp op) {
  HANDLE_BINARY_OP_INT(op, +, "arith.addi");
}

void Interpreter::visit(arith::AddFOp op) {
  HANDLE_BINARY_OP_FLOAT(op, +, "arith.addf");
}

void Interpreter::visit(arith::SubIOp op) {
  HANDLE_BINARY_OP_INT(op, -, "arith.subi");
}

void Interpreter::visit(arith::SubFOp op) {
  HANDLE_BINARY_OP_FLOAT(op, -, "arith.subf");
}

void Interpreter::visit(arith::MulIOp op) {
  HANDLE_BINARY_OP_INT(op, *, "arith.muli");
}

void Interpreter::visit(arith::MulFOp op) {
  HANDLE_BINARY_OP_FLOAT(op, *, "arith.mulf");
}

void Interpreter::visit(arith::DivSIOp op) {
  HANDLE_BINARY_OP_INT(op, /, "arith.divsi");
}

void Interpreter::visit(arith::RemSIOp op) {
  HANDLE_BINARY_OP_INT(op, %, "arith.remsi");
}

void Interpreter::visit(arith::AndIOp op) {
  HANDLE_BINARY_OP_INT(op, &, "arith.andi");
}

void Interpreter::visit(arith::MaxSIOp op) {
  HANDLE_BINARY_OP_INT_FUNC(op, std::max, "arith.maxsi");
}

void Interpreter::visit(arith::MinSIOp op) {
  HANDLE_BINARY_OP_INT_FUNC(op, std::min, "arith.minsi");
}

void Interpreter::visit(arith::CmpIOp op) {
  auto lhsType = op.getLhs().getType();
  auto cmpFunc = [&](auto a, auto b) -> int {
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:
        return a == b ? 1 : 0;
      case arith::CmpIPredicate::ne:
        return a != b ? 1 : 0;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        return a < b ? 1 : 0;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        return a <= b ? 1 : 0;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        return a > b ? 1 : 0;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        return a >= b ? 1 : 0;
    }
    return 0;
  };

  if (lhsType.isInteger()) {
    intValues[op.getResult()] =
        cmpFunc(intValues.at(op.getLhs()), intValues.at(op.getRhs()));
  } else if (lhsType.isF32()) {
    intValues[op.getResult()] =
        cmpFunc(floatValues.at(op.getLhs()), floatValues.at(op.getRhs()));
  } else if (lhsType.isF64()) {
    intValues[op.getResult()] =
        cmpFunc(doubleValues.at(op.getLhs()), doubleValues.at(op.getRhs()));
  } else if (lhsType.isIndex()) {
    intValues[op.getResult()] =
        cmpFunc(intValues.at(op.getLhs()), intValues.at(op.getRhs()));
  } else {
    op->emitError() << "Unsupported type for arith.cmpi: " << lhsType << "\n";
  }
}

void Interpreter::visit(arith::SelectOp op) {
  int cond = intValues.at(op.getCondition());
  bool condBool = (cond != 0);

  auto resultType = op.getResult().getType();
  if (resultType.isInteger()) {
    intValues[op.getResult()] = condBool ? intValues.at(op.getTrueValue())
                                         : intValues.at(op.getFalseValue());
  } else if (resultType.isF32()) {
    floatValues[op.getResult()] = condBool ? floatValues.at(op.getTrueValue())
                                           : floatValues.at(op.getFalseValue());
  } else if (resultType.isF64()) {
    doubleValues[op.getResult()] = condBool
                                       ? doubleValues.at(op.getTrueValue())
                                       : doubleValues.at(op.getFalseValue());
  } else if (resultType.isIndex()) {
    intValues[op.getResult()] = condBool ? intValues.at(op.getTrueValue())
                                         : intValues.at(op.getFalseValue());
  } else {
    op->emitError() << "Unsupported type for arith.select: " << resultType
                    << "\n";
  }
}

void Interpreter::visit(arith::FloorDivSIOp op) {
  HANDLE_BINARY_OP_INT_FUNC(op, floorDivInt, "arith.floordivsi");
}

void Interpreter::visit(arith::ExtFOp op) {
  auto inType = op.getIn().getType();
  auto outType = op.getOut().getType();

  if (auto tensorType = dyn_cast<RankedTensorType>(inType)) {
    // Handle tensor case
    auto elemType = tensorType.getElementType();
    auto outElemType = cast<RankedTensorType>(outType).getElementType();

    if (elemType.isF32() && outElemType.isF64()) {
      // Convert float vector to double vector
      auto inVec = floatVectors.at(op.getIn());
      auto outVec =
          std::make_shared<std::vector<double>>(inVec->begin(), inVec->end());
      doubleVectors[op.getResult()] = outVec;
    } else if (elemType.isF32()) {
      floatVectors[op.getResult()] = floatVectors.at(op.getIn());
    } else {
      doubleVectors[op.getResult()] = doubleVectors.at(op.getIn());
    }
  } else {
    // Scalar case
    if (inType.isF32() && outType.isF64()) {
      doubleValues[op.getResult()] =
          static_cast<double>(floatValues.at(op.getIn()));
    } else if (inType.isF32()) {
      floatValues[op.getResult()] = floatValues.at(op.getIn());
    } else {
      doubleValues[op.getResult()] = doubleValues.at(op.getIn());
    }
  }
}

int Interpreter::getFlattenedTensorIndex(Value tensor, ValueRange indices) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto shape = tensorType.getShape();
  int accum = intValues.at(indices[0]);
  for (size_t i = 1; i < shape.size(); ++i) {
    accum = accum * shape[i] + intValues.at(indices[i]);
  }
  return accum;
}

void Interpreter::visit(tensor::EmptyOp op) {
  auto tensorType = op.getResult().getType();
  auto numElements = tensorType.getNumElements();
  auto elementType = tensorType.getElementType();

  if (elementType.isInteger()) {
    intVectors[op.getResult()] =
        std::make_shared<std::vector<int>>(numElements);
  } else if (elementType.isF32()) {
    floatVectors[op.getResult()] =
        std::make_shared<std::vector<float>>(numElements);
  } else if (elementType.isF64()) {
    doubleVectors[op.getResult()] =
        std::make_shared<std::vector<double>>(numElements);
  } else if (isa<lwe::LWEPlaintextType>(elementType)) {
    plaintextVectors[op.getResult()] =
        std::make_shared<std::vector<PlaintextT>>(numElements);
  } else {
    ciphertextVectors[op.getResult()] =
        std::make_shared<std::vector<CiphertextT>>(numElements);
  }
}

void Interpreter::visit(tensor::ExtractOp op) {
  int index = getFlattenedTensorIndex(op.getTensor(), op.getIndices());
  auto tensorType = cast<RankedTensorType>(op.getTensor().getType());
  auto elemType = tensorType.getElementType();

  if (elemType.isInteger()) {
    intValues[op.getResult()] = (*intVectors.at(op.getTensor()))[index];
  } else if (elemType.isF32()) {
    floatValues[op.getResult()] = (*floatVectors.at(op.getTensor()))[index];
  } else if (elemType.isF64()) {
    doubleValues[op.getResult()] = (*doubleVectors.at(op.getTensor()))[index];
  } else if (isa<lwe::LWEPlaintextType>(elemType)) {
    plaintexts[op.getResult()] = (*plaintextVectors.at(op.getTensor()))[index];
  } else {
    ciphertexts[op.getResult()] =
        (*ciphertextVectors.at(op.getTensor()))[index];
  }
}

void Interpreter::visit(tensor::InsertOp op) {
  int index = getFlattenedTensorIndex(op.getDest(), op.getIndices());
  auto tensorType = cast<RankedTensorType>(op.getDest().getType());
  auto elemType = tensorType.getElementType();

  // Check if we can modify the tensor in-place (no copy needed)
  bool canModifyInPlace = liveness.isDeadAfter(op.getDest(), op);

  if (elemType.isInteger(32) || elemType.isInteger(64)) {
    auto srcVec = intVectors.at(op.getDest());
    auto vec =
        canModifyInPlace ? srcVec : std::make_shared<std::vector<int>>(*srcVec);
    (*vec)[index] = intValues.at(op.getScalar());
    intVectors[op.getResult()] = vec;
  } else if (elemType.isF32()) {
    auto srcVec = floatVectors.at(op.getDest());
    auto vec = canModifyInPlace ? srcVec
                                : std::make_shared<std::vector<float>>(*srcVec);
    (*vec)[index] = floatValues.at(op.getScalar());
    floatVectors[op.getResult()] = vec;
  } else if (elemType.isF64()) {
    auto srcVec = doubleVectors.at(op.getDest());
    auto vec = canModifyInPlace
                   ? srcVec
                   : std::make_shared<std::vector<double>>(*srcVec);
    (*vec)[index] = doubleValues.at(op.getScalar());
    doubleVectors[op.getResult()] = vec;
  } else if (isa<lwe::LWEPlaintextType>(elemType)) {
    auto srcVec = plaintextVectors.at(op.getDest());
    auto vec = canModifyInPlace
                   ? srcVec
                   : std::make_shared<std::vector<PlaintextT>>(*srcVec);
    (*vec)[index] = plaintexts.at(op.getScalar());
    plaintextVectors[op.getResult()] = vec;
  } else {
    auto srcVec = ciphertextVectors.at(op.getDest());
    auto vec = canModifyInPlace
                   ? srcVec
                   : std::make_shared<std::vector<CiphertextT>>(*srcVec);
    (*vec)[index] = ciphertexts.at(op.getScalar());
    ciphertextVectors[op.getResult()] = vec;
  }
}

void Interpreter::visit(tensor::SplatOp op) {
  auto tensorType = op.getResult().getType();
  auto numElements = tensorType.getNumElements();
  auto elemType = tensorType.getElementType();

  if (elemType.isInteger(32) || elemType.isInteger(64)) {
    int val = intValues.at(op.getInput());
    intVectors[op.getResult()] =
        std::make_shared<std::vector<int>>(numElements, val);
  } else if (elemType.isF32()) {
    float val = floatValues.at(op.getInput());
    floatVectors[op.getResult()] =
        std::make_shared<std::vector<float>>(numElements, val);
  } else if (elemType.isF64()) {
    double val = doubleValues.at(op.getInput());
    doubleVectors[op.getResult()] =
        std::make_shared<std::vector<double>>(numElements, val);
  }
}

void Interpreter::visit(tensor::FromElementsOp op) {
  auto elements = op.getElements();
  if (elements.empty()) {
    op.emitError("FromElementsOp requires at least one element");
    return;
  }

  auto tensorType = op.getResult().getType();
  auto elemType = tensorType.getElementType();

  if (elemType.isInteger(32) || elemType.isInteger(64)) {
    auto result = std::make_shared<std::vector<int>>(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      (*result)[i] = intValues.at(elements[i]);
    }
    intVectors[op.getResult()] = result;
  } else if (elemType.isF32()) {
    auto result = std::make_shared<std::vector<float>>(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      (*result)[i] = floatValues.at(elements[i]);
    }
    floatVectors[op.getResult()] = result;
  } else if (elemType.isF64()) {
    auto result = std::make_shared<std::vector<double>>(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      (*result)[i] = doubleValues.at(elements[i]);
    }
    doubleVectors[op.getResult()] = result;
  } else if (isa<lwe::LWEPlaintextType>(elemType)) {
    auto result = std::make_shared<std::vector<PlaintextT>>(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      (*result)[i] = plaintexts.at(elements[i]);
    }
    plaintextVectors[op.getResult()] = result;
  } else {
    auto result = std::make_shared<std::vector<CiphertextT>>(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      (*result)[i] = ciphertexts.at(elements[i]);
    }
    ciphertextVectors[op.getResult()] = result;
  }
}

void Interpreter::visit(tensor::ConcatOp op) {
  auto inputs = op.getInputs();
  auto tensorType = op.getResult().getType();
  auto elemType = tensorType.getElementType();

  if (elemType.isInteger(32) || elemType.isInteger(64)) {
    // Calculate total size and reserve
    size_t totalSize = 0;
    for (auto input : inputs) {
      totalSize += intVectors.at(input)->size();
    }
    auto result = std::make_shared<std::vector<int>>();
    result->reserve(totalSize);
    for (auto input : inputs) {
      const auto& vec = *intVectors.at(input);
      result->insert(result->end(), vec.begin(), vec.end());
    }
    intVectors[op.getResult()] = result;
  } else if (elemType.isF32()) {
    size_t totalSize = 0;
    for (auto input : inputs) {
      totalSize += floatVectors.at(input)->size();
    }
    auto result = std::make_shared<std::vector<float>>();
    result->reserve(totalSize);
    for (auto input : inputs) {
      const auto& vec = *floatVectors.at(input);
      result->insert(result->end(), vec.begin(), vec.end());
    }
    floatVectors[op.getResult()] = result;
  } else if (elemType.isF64()) {
    size_t totalSize = 0;
    for (auto input : inputs) {
      totalSize += doubleVectors.at(input)->size();
    }
    auto result = std::make_shared<std::vector<double>>();
    result->reserve(totalSize);
    for (auto input : inputs) {
      const auto& vec = *doubleVectors.at(input);
      result->insert(result->end(), vec.begin(), vec.end());
    }
    doubleVectors[op.getResult()] = result;
  } else if (isa<lwe::LWEPlaintextType>(elemType)) {
    size_t totalSize = 0;
    for (auto input : inputs) {
      totalSize += plaintextVectors.at(input)->size();
    }
    auto result = std::make_shared<std::vector<PlaintextT>>();
    result->reserve(totalSize);
    for (auto input : inputs) {
      const auto& vec = *plaintextVectors.at(input);
      result->insert(result->end(), vec.begin(), vec.end());
    }
    plaintextVectors[op.getResult()] = result;
  } else {
    size_t totalSize = 0;
    for (auto input : inputs) {
      totalSize += ciphertextVectors.at(input)->size();
    }
    auto result = std::make_shared<std::vector<CiphertextT>>();
    result->reserve(totalSize);
    for (auto input : inputs) {
      const auto& vec = *ciphertextVectors.at(input);
      result->insert(result->end(), vec.begin(), vec.end());
    }
    ciphertextVectors[op.getResult()] = result;
  }
}

void Interpreter::visit(tensor::CollapseShapeOp op) {
  // Just copy the shared_ptr (shape metadata doesn't affect our flat storage)
  auto srcType = cast<RankedTensorType>(op.getSrc().getType());
  auto elemType = srcType.getElementType();

  if (elemType.isInteger(32) || elemType.isInteger(64)) {
    intVectors[op.getResult()] = intVectors.at(op.getSrc());
  } else if (elemType.isF32()) {
    floatVectors[op.getResult()] = floatVectors.at(op.getSrc());
  } else if (elemType.isF64()) {
    doubleVectors[op.getResult()] = doubleVectors.at(op.getSrc());
  } else if (isa<lwe::LWEPlaintextType>(elemType)) {
    plaintextVectors[op.getResult()] = plaintextVectors.at(op.getSrc());
  } else {
    ciphertextVectors[op.getResult()] = ciphertextVectors.at(op.getSrc());
  }
}

void Interpreter::visit(tensor::ExpandShapeOp op) {
  // Same as CollapseShapeOp
  auto srcType = cast<RankedTensorType>(op.getSrc().getType());
  auto elemType = srcType.getElementType();

  if (elemType.isInteger(32) || elemType.isInteger(64)) {
    intVectors[op.getResult()] = intVectors.at(op.getSrc());
  } else if (elemType.isF32()) {
    floatVectors[op.getResult()] = floatVectors.at(op.getSrc());
  } else if (elemType.isF64()) {
    doubleVectors[op.getResult()] = doubleVectors.at(op.getSrc());
  } else if (isa<lwe::LWEPlaintextType>(elemType)) {
    plaintextVectors[op.getResult()] = plaintextVectors.at(op.getSrc());
  } else {
    ciphertextVectors[op.getResult()] = ciphertextVectors.at(op.getSrc());
  }
}

void Interpreter::visit(tensor::ExtractSliceOp op) {
  auto srcType = cast<RankedTensorType>(op.getSource().getType());
  auto elemType = srcType.getElementType();

  auto offsets = op.getStaticOffsets();
  auto sizes = op.getStaticSizes();
  auto strides = op.getStaticStrides();

  auto sourceType = cast<RankedTensorType>(op.getSource().getType());
  auto sourceShape = sourceType.getShape();

  // Calculate total number of elements to extract
  int64_t totalElements = 1;
  for (int64_t size : sizes) {
    totalElements *= size;
  }

  // For multi-dimensional slices, we need to compute which elements to extract
  // from the flattened source tensor
  auto extractElement = [&](int64_t flatResultIndex) -> int64_t {
    // Convert flat result index to multi-dimensional result indices
    std::vector<int64_t> resultIndices(sizes.size());
    int64_t remaining = flatResultIndex;
    for (int i = sizes.size() - 1; i >= 0; --i) {
      resultIndices[i] = remaining % sizes[i];
      remaining /= sizes[i];
    }

    // Convert to source indices using offsets and strides
    std::vector<int64_t> sourceIndices(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
      sourceIndices[i] = offsets[i] + resultIndices[i] * strides[i];
    }

    // Convert source indices to flat index
    int64_t flatSourceIndex = sourceIndices[0];
    for (size_t i = 1; i < sourceIndices.size(); ++i) {
      flatSourceIndex = flatSourceIndex * sourceShape[i] + sourceIndices[i];
    }
    return flatSourceIndex;
  };

  if (elemType.isInteger()) {
    auto result = std::vector<int>(totalElements);
    const auto& srcVec = *intVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      result[i] = srcVec[extractElement(i)];
    }
    intVectors[op.getResult()] =
        std::make_shared<std::vector<int>>(std::move(result));
  } else if (elemType.isF32()) {
    auto result = std::vector<float>(totalElements);
    const auto& srcVec = *floatVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      result[i] = srcVec[extractElement(i)];
    }
    floatVectors[op.getResult()] =
        std::make_shared<std::vector<float>>(std::move(result));
  } else if (elemType.isF64()) {
    auto result = std::vector<double>(totalElements);
    const auto& srcVec = *doubleVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      result[i] = srcVec[extractElement(i)];
    }
    doubleVectors[op.getResult()] =
        std::make_shared<std::vector<double>>(std::move(result));
  } else if (isa<lwe::LWEPlaintextType>(elemType)) {
    auto result = std::vector<PlaintextT>(totalElements);
    const auto& srcVec = *plaintextVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      result[i] = srcVec[extractElement(i)];
    }
    plaintextVectors[op.getResult()] =
        std::make_shared<std::vector<PlaintextT>>(std::move(result));
  } else if (isa<lwe::LWECiphertextType>(elemType)) {
    auto result = std::vector<CiphertextT>(totalElements);
    const auto& srcVec = *ciphertextVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      result[i] = srcVec[extractElement(i)];
    }
    ciphertextVectors[op.getResult()] =
        std::make_shared<std::vector<CiphertextT>>(std::move(result));
  } else {
    op.emitError("Unsupported tensor type in ExtractSliceOp\n");
  }
}

void Interpreter::visit(tensor::InsertSliceOp op) {
  auto offsets = op.getStaticOffsets();
  auto sizes = op.getStaticSizes();
  auto strides = op.getStaticStrides();

  auto destType = cast<RankedTensorType>(op.getDest().getType());
  auto destShape = destType.getShape();

  // Calculate total number of elements to insert
  int64_t totalElements = 1;
  for (int64_t size : sizes) {
    totalElements *= size;
  }

  // For multi-dimensional slices, we need to compute which dest elements to
  // update
  auto insertElement = [&](int64_t flatSourceIndex) -> int64_t {
    // Convert flat source index to multi-dimensional source indices
    std::vector<int64_t> sourceIndices(sizes.size());
    int64_t remaining = flatSourceIndex;
    for (int i = sizes.size() - 1; i >= 0; --i) {
      sourceIndices[i] = remaining % sizes[i];
      remaining /= sizes[i];
    }

    // Convert to dest indices using offsets and strides
    std::vector<int64_t> destIndices(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) {
      destIndices[i] = offsets[i] + sourceIndices[i] * strides[i];
    }

    // Convert dest indices to flat index
    int64_t flatDestIndex = destIndices[0];
    for (size_t i = 1; i < destIndices.size(); ++i) {
      flatDestIndex = flatDestIndex * destShape[i] + destIndices[i];
    }
    return flatDestIndex;
  };

  bool canModifyInPlace = liveness.isDeadAfter(op.getDest(), op);

  if (auto elemType = destType.getElementType(); elemType.isInteger()) {
    auto srcDestVec = intVectors.at(op.getDest());
    auto destVec = canModifyInPlace
                       ? srcDestVec
                       : std::make_shared<std::vector<int>>(*srcDestVec);
    const auto& srcVec = *intVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      (*destVec)[insertElement(i)] = srcVec[i];
    }
    intVectors[op.getResult()] = std::move(destVec);
  } else if (elemType.isF32()) {
    auto srcDestVec = floatVectors.at(op.getDest());
    auto destVec = canModifyInPlace
                       ? srcDestVec
                       : std::make_shared<std::vector<float>>(*srcDestVec);
    const auto& srcVec = *floatVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      (*destVec)[insertElement(i)] = srcVec[i];
    }
    floatVectors[op.getResult()] = std::move(destVec);
  } else if (elemType.isF64()) {
    auto srcDestVec = doubleVectors.at(op.getDest());
    auto destVec = canModifyInPlace
                       ? srcDestVec
                       : std::make_shared<std::vector<double>>(*srcDestVec);
    const auto& srcVec = *doubleVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      (*destVec)[insertElement(i)] = srcVec[i];
    }
    doubleVectors[op.getResult()] = std::move(destVec);
  } else if (isa<lwe::LWEPlaintextType>(elemType)) {
    auto srcDestVec = plaintextVectors.at(op.getDest());
    auto destVec = canModifyInPlace
                       ? srcDestVec
                       : std::make_shared<std::vector<PlaintextT>>(*srcDestVec);
    const auto& srcVec = *plaintextVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      (*destVec)[insertElement(i)] = srcVec[i];
    }
    plaintextVectors[op.getResult()] = std::move(destVec);
  } else if (isa<lwe::LWECiphertextType>(elemType)) {
    auto srcDestVec = ciphertextVectors.at(op.getDest());
    auto destVec =
        canModifyInPlace
            ? srcDestVec
            : std::make_shared<std::vector<CiphertextT>>(*srcDestVec);
    const auto& srcVec = *ciphertextVectors.at(op.getSource());
    for (int64_t i = 0; i < totalElements; ++i) {
      (*destVec)[insertElement(i)] = srcVec[i];
    }
    ciphertextVectors[op.getResult()] = std::move(destVec);
  } else {
    op.emitError("Unsupported tensor type in InsertSliceOp\n");
  }
}

void Interpreter::visit(linalg::BroadcastOp op) {
  // BroadcastOp copies the input op into a new tensor by adding the specified
  // dims.
  auto resultType = cast<RankedTensorType>(op->getResults()[0].getType());
  auto resultShape = resultType.getShape();
  auto inputType = cast<RankedTensorType>(op.getInput().getType());
  auto inputShape = inputType.getShape();
  auto numOutputElements = resultType.getNumElements();

  auto calculate = [&](const auto& inputVec, auto& outputVec) {
    // Create a map of the broadcast dimensions.
    DenseMap<int64_t, bool> broadcastDims;
    for (const auto& dim : op.getDimensions()) {
      broadcastDims[dim] = true;
    }

    // Pre-allocate temporary index vectors outside the loop
    std::vector<int64_t> outputIndices(resultShape.size());
    std::vector<int64_t> inputIndices(inputShape.size());

    // Iterate over the output tensor's elements
    for (int64_t i = 0; i < numOutputElements; ++i) {
      // Calculate the multi-dimensional index in the output tensor by
      // unflattening the index in the output shape.
      int64_t temp = i;
      for (int d = resultShape.size() - 1; d >= 0; --d) {
        if (resultShape[d] > 0) {
          outputIndices[d] = temp % resultShape[d];
          temp /= resultShape[d];
        } else {
          outputIndices[d] = 0;
        }
      }

      // Calculate the multi-dimensional index in the input tensor.
      int64_t inputDim = 0;
      for (size_t d = 0; d < resultShape.size(); ++d) {
        if (!broadcastDims.contains(d)) {
          // If the output dimension is not broadcast, then the input and
          // output dimensions are the same.
          inputIndices[inputDim] = outputIndices[d];
          inputDim++;
        }
      }

      // Calculate the flattened index in the input tensor
      int64_t inputFlatIndex = 0;
      if (!inputShape.empty()) {
        inputFlatIndex = inputIndices[0];
        for (size_t d = 1; d < inputShape.size(); ++d) {
          inputFlatIndex = inputFlatIndex * inputShape[d] + inputIndices[d];
        }
      }

      outputVec[i] = inputVec[inputFlatIndex];
    }
  };

  if (inputType.getElementType().isInteger()) {
    auto inputVec = *intVectors.at(op.getInput());
    auto outputVec = std::vector<int>(numOutputElements);
    calculate(inputVec, outputVec);
    intVectors[op->getResults()[0]] =
        std::make_shared<std::vector<int>>(std::move(outputVec));
  } else if (inputType.getElementType().isF32()) {
    auto inputVec = *floatVectors.at(op.getInput());
    auto outputVec = std::vector<float>(numOutputElements);
    calculate(inputVec, outputVec);
    floatVectors[op->getResults()[0]] =
        std::make_shared<std::vector<float>>(std::move(outputVec));
  } else if (inputType.getElementType().isF64()) {
    auto inputVec = *doubleVectors.at(op.getInput());
    auto outputVec = std::vector<double>(numOutputElements);
    calculate(inputVec, outputVec);
    doubleVectors[op->getResults()[0]] =
        std::make_shared<std::vector<double>>(std::move(outputVec));
  } else {
    op.emitError("Unsupported tensor type in BroadcastOp\n");
  }
}

// SCF and Affine ops
void Interpreter::visit(scf::YieldOp op) {
  // YieldOp is handled specially within loop bodies
  // The operands are already in env and will be retrieved by the loop op
}

void Interpreter::visit(affine::AffineYieldOp op) {
  // AffineYieldOp is handled specially within loop bodies
  // The operands are already in env and will be retrieved by the loop op
}

void Interpreter::visit(scf::ForOp op) {
  int lowerBound = intValues.at(op.getLowerBound());
  int upperBound = intValues.at(op.getUpperBound());
  int step = intValues.at(op.getStep());

  // Initialize iter args with initial values
  std::vector<TypedCppValue> iterArgs;
  iterArgs.reserve(op.getInitArgs().size());
  for (auto initArg : op.getInitArgs()) {
    iterArgs.push_back(loadTypedValue(initArg));
  }

  // Cache operation visitors to avoid dispatch overhead in hot loop
  std::vector<std::function<void()>> cachedOps;
  scf::YieldOp yieldOp = nullptr;
  for (auto& bodyOp : op.getBody()->getOperations()) {
    if (auto yield = dyn_cast<scf::YieldOp>(&bodyOp)) {
      yieldOp = yield;
    } else {
      Operation* opPtr = &bodyOp;  // Capture pointer, not reference to loop var
      cachedOps.push_back([this, opPtr]() { visit(opPtr); });
    }
  }

  // Execute the loop
  for (int i = lowerBound; i < upperBound; i += step) {
    intValues[op.getInductionVar()] = i;
    for (auto [blockArg, iterArg] :
         llvm::zip(op.getRegionIterArgs(), iterArgs)) {
      storeTypedValue(blockArg, iterArg);
    }

    // Execute loop body
    for (auto& cachedOp : cachedOps) {
      cachedOp();
    }

    // Collect yield results
    std::vector<TypedCppValue> yieldResults;
    yieldResults.reserve(iterArgs.size());
    if (yieldOp) {
      for (auto yieldOperand : yieldOp.getOperands()) {
        yieldResults.push_back(loadTypedValue(yieldOperand));
      }
    }

    iterArgs = std::move(yieldResults);
  }

  // Store final results
  for (auto [result, finalValue] : llvm::zip(op.getResults(), iterArgs)) {
    storeTypedValue(result, std::move(finalValue));
  }
}

void Interpreter::visit(scf::IfOp op) {
  int condition = intValues.at(op.getCondition());
  bool condBool = (condition != 0);
  std::vector<TypedCppValue> results;

  if (condBool) {
    for (auto& bodyOp : op.getThenRegion().front().getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
        for (auto yieldOperand : yieldOp.getOperands()) {
          results.push_back(loadTypedValue(yieldOperand));
        }
      } else {
        visit(&bodyOp);
      }
    }
  } else if (!op.getElseRegion().empty()) {
    for (auto& bodyOp : op.getElseRegion().front().getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
        for (auto yieldOperand : yieldOp.getOperands()) {
          results.push_back(loadTypedValue(yieldOperand));
        }
      } else {
        visit(&bodyOp);
      }
    }
  }

  for (auto [result, value] : llvm::zip(op.getResults(), results)) {
    storeTypedValue(result, std::move(value));
  }
}

void Interpreter::visit(affine::AffineForOp op) {
  // Get loop bounds from affine maps
  auto lowerBoundMap = op.getLowerBoundMap();
  auto upperBoundMap = op.getUpperBoundMap();

  // For simplicity, assume constant bounds (map with no inputs)
  if (lowerBoundMap.getNumInputs() != 0 || upperBoundMap.getNumInputs() != 0) {
    op.emitError("AffineForOp with non-constant bounds not yet supported");
  }

  int64_t lowerBound = lowerBoundMap.getSingleConstantResult();
  int64_t upperBound = upperBoundMap.getSingleConstantResult();
  int64_t step = op.getStep().getSExtValue();

  // Initialize iter args with initial values
  std::vector<TypedCppValue> iterArgs;
  iterArgs.reserve(op.getInits().size());
  for (auto initArg : op.getInits()) {
    iterArgs.push_back(loadTypedValue(initArg));
  }

  // Cache operation visitors to avoid dispatch overhead in hot loop
  std::vector<std::function<void()>> cachedOps;
  affine::AffineYieldOp yieldOp = nullptr;
  for (auto& bodyOp : op.getBody()->getOperations()) {
    if (auto yield = dyn_cast<affine::AffineYieldOp>(&bodyOp)) {
      yieldOp = yield;
    } else {
      Operation* opPtr = &bodyOp;  // Capture pointer, not reference to loop var
      cachedOps.push_back([this, opPtr]() { visit(opPtr); });
    }
  }

  // Execute the loop
  for (int64_t i = lowerBound; i < upperBound; i += step) {
    // Set up induction variable and iter args in env
    eraseValue(op.getInductionVar());
    storeTypedValue(op.getInductionVar(), TypedCppValue(static_cast<int>(i)));

    for (auto [blockArg, iterArg] :
         llvm::zip(op.getRegionIterArgs(), iterArgs)) {
      eraseValue(blockArg);
      storeTypedValue(blockArg, iterArg);
    }

    // Execute cached operations
    for (auto& cachedOp : cachedOps) {
      cachedOp();
    }

    // Collect yield results
    std::vector<TypedCppValue> yieldResults;
    yieldResults.reserve(iterArgs.size());
    if (yieldOp) {
      for (auto yieldOperand : yieldOp.getOperands()) {
        yieldResults.push_back(loadTypedValue(yieldOperand));
      }
    }

    // Update iter args for next iteration
    iterArgs = std::move(yieldResults);
  }

  // Store final results (move to avoid copying)
  for (auto [result, finalValue] : llvm::zip(op.getResults(), iterArgs)) {
    storeTypedValue(result, std::move(finalValue));
  }
}

#ifdef OPENFHE_ENABLE_TIMING
#include <iomanip>
#include <iostream>
#include <vector>

void Interpreter::printTimingResults() {
  double totalTime = 0;
  for (const auto& entry : timingResults) {
    totalTime += entry.second.totalTime.count();
  }

  struct TimingInfo {
    std::string operation;
    double totalTime;
    int count;
    double percentage;
  };

  std::vector<TimingInfo> sortedResults;
  for (const auto& entry : timingResults) {
    sortedResults.push_back(
        {entry.first, entry.second.totalTime.count(), entry.second.count,
         (entry.second.totalTime.count() / totalTime) * 100});
  }

  std::sort(sortedResults.begin(), sortedResults.end(),
            [](const TimingInfo& a, const TimingInfo& b) {
              return a.percentage > b.percentage;
            });

  std::cout << "--- Timing Results ---\n";
  std::cout << std::left << std::setw(30) << "Operation" << std::setw(20)
            << "Total Time (s)" << std::setw(20) << "Total Time (%)"
            << std::setw(20) << "Count" << "Average Latency (s)\n";
  for (const auto& entry : sortedResults) {
    std::cout << std::left << std::setw(30) << entry.operation << std::setw(20)
              << entry.totalTime << std::setw(20) << entry.percentage
              << std::setw(20) << entry.count << (entry.totalTime / entry.count)
              << "\n";
  }
}
#endif

// Macro for handling binary operations on integer types
#define HANDLE_CT_CT_BINOP(op, opName, evalMethod)                         \
  do {                                                                     \
    auto cc = cryptoContexts.at((op).getCryptoContext());                  \
    auto lhsCt = ciphertexts.at((op).getLhs());                            \
    auto rhsCt = ciphertexts.at((op).getRhs());                            \
    TIME_OPERATION("Add", (op).getOutput(), cc->evalMethod(lhsCt, rhsCt)); \
  } while (0)

// OpenFHE ct-ct binary operations
void Interpreter::visit(AddOp op) { HANDLE_CT_CT_BINOP(op, "Add", EvalAdd); }
void Interpreter::visit(SubOp op) { HANDLE_CT_CT_BINOP(op, "Sub", EvalSub); }
void Interpreter::visit(MulOp op) { HANDLE_CT_CT_BINOP(op, "Mul", EvalMult); }
void Interpreter::visit(MulNoRelinOp op) {
  HANDLE_CT_CT_BINOP(op, "MulNoRelin", EvalMultNoRelin);
}

void Interpreter::visit(AddPlainOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto lhsVal = op.getLhs();
  auto rhsVal = op.getRhs();

  if (isa<lwe::LWECiphertextType>(lhsVal.getType()) &&
      isa<lwe::LWEPlaintextType>(rhsVal.getType())) {
    auto lhsCt = ciphertexts.at(lhsVal);
    auto rhsPt = plaintexts.at(rhsVal);
    TIME_OPERATION("AddPlain", op.getOutput(), cc->EvalAdd(lhsCt, rhsPt));
  } else if (isa<lwe::LWEPlaintextType>(lhsVal.getType()) &&
             isa<lwe::LWECiphertextType>(rhsVal.getType())) {
    auto lhsPt = plaintexts.at(lhsVal);
    auto rhsCt = ciphertexts.at(rhsVal);
    TIME_OPERATION("AddPlain", op.getOutput(), cc->EvalAdd(lhsPt, rhsCt));
  } else {
    op.emitError("AddPlainOp requires ciphertext and plaintext");
  }
}

void Interpreter::visit(SubPlainOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto lhsVal = op.getLhs();
  auto rhsVal = op.getRhs();

  // Check which is ciphertext and which is plaintext
  if (isa<lwe::LWECiphertextType>(lhsVal.getType()) &&
      isa<lwe::LWEPlaintextType>(rhsVal.getType())) {
    auto lhsCt = ciphertexts.at(lhsVal);
    auto rhsPt = plaintexts.at(rhsVal);
    TIME_OPERATION("SubPlain", op.getOutput(), cc->EvalSub(lhsCt, rhsPt));
  } else if (isa<lwe::LWEPlaintextType>(lhsVal.getType()) &&
             isa<lwe::LWECiphertextType>(rhsVal.getType())) {
    auto lhsPt = plaintexts.at(lhsVal);
    auto rhsCt = ciphertexts.at(rhsVal);
    TIME_OPERATION("SubPlain", op.getOutput(), cc->EvalSub(lhsPt, rhsCt));
  } else {
    op.emitError("SubPlainOp requires ciphertext and plaintext");
  }
}

void Interpreter::visit(MulPlainOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getCiphertext());
  auto pt = plaintexts.at(op.getPlaintext());
  TIME_OPERATION("MulPlain", op.getOutput(), cc->EvalMult(ct, pt));
}

void Interpreter::visit(MulConstOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getCiphertext());
  auto constVal = intValues.at(op.getConstant());
  TIME_OPERATION("MulConst", op.getOutput(), cc->EvalMult(ct, constVal));
}

// OpenFHE unary operations
// Macro for handling binary operations on integer types
#define HANDLE_CT_UNARY(op, opName, evalMethod)                       \
  do {                                                                \
    auto cc = cryptoContexts.at((op).getCryptoContext());             \
    auto inputCt = ciphertexts.at((op).getCiphertext());              \
    TIME_OPERATION("Add", (op).getOutput(), cc->evalMethod(inputCt)); \
  } while (0)

void Interpreter::visit(NegateOp op) {
  HANDLE_CT_UNARY(op, "Negate", EvalNegate);
}

void Interpreter::visit(SquareOp op) {
  HANDLE_CT_UNARY(op, "Square", EvalSquare);
}

void Interpreter::visit(RelinOp op) {
  HANDLE_CT_UNARY(op, "Relin", Relinearize);
}

void Interpreter::visit(ModReduceOp op) {
  HANDLE_CT_UNARY(op, "ModReduce", ModReduce);
}

void Interpreter::visit(LevelReduceOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getCiphertext());
  auto levelToDrop = op.getLevelToDrop();
  TIME_OPERATION("LevelReduce", op.getOutput(),
                 cc->LevelReduce(ct, nullptr, levelToDrop));
}

void Interpreter::visit(RotOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getCiphertext());
  auto index = op.getIndex().getValue().getSExtValue();
  TIME_OPERATION("Rot", op.getOutput(), cc->EvalRotate(ct, index));
}

void Interpreter::visit(AutomorphOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getCiphertext());
  auto evalKey = evalKeys.at(op.getEvalKey());
  // Note: AutomorphOp requires building a map with the eval key
  // For simplicity, we'll use index 0 as in the emitter
  std::map<uint32_t, EvalKeyT> evalKeyMap = {{0, evalKey}};
  TIME_OPERATION("Automorph", op.getOutput(),
                 cc->EvalAutomorphism(ct, 0, evalKeyMap));
}

void Interpreter::visit(KeySwitchOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getCiphertext());
  auto key = evalKeys.at(op.getEvalKey());
  TIME_OPERATION("KeySwitch", op.getOutput(), cc->KeySwitch(ct, key));
}

void Interpreter::visit(BootstrapOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getCiphertext());
  TIME_OPERATION("Bootstrap", op.getOutput(), cc->EvalBootstrap(ct));
}

// OpenFHE encryption/decryption operations
void Interpreter::visit(EncryptOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto pt = plaintexts.at(op.getPlaintext());
  // Note: EncryptOp takes an encryption key which could be public or private
  // For now, we'll need to handle both cases
  if (publicKeys.find(op.getEncryptionKey()) != publicKeys.end()) {
    auto key = publicKeys.at(op.getEncryptionKey());
    TIME_OPERATION("Encrypt", op.getCiphertext(), cc->Encrypt(key, pt));
  } else {
    auto key = privateKeys.at(op.getEncryptionKey());
    TIME_OPERATION("Encrypt", op.getCiphertext(), cc->Encrypt(key, pt));
  }
}

void Interpreter::visit(DecryptOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ciphertext = ciphertexts.at(op.getCiphertext());
  auto key = privateKeys.at(op.getPrivateKey());
  PlaintextT plaintext;
  TIME_OPERATION_VOID("Decrypt", cc->Decrypt(key, ciphertext, &plaintext));
  plaintexts[op.getPlaintext()] = plaintext;
}

void Interpreter::visit(MakePackedPlaintextOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  const auto& vec = *intVectors.at(op.getValue());
  std::vector<int64_t> vec64(vec.begin(), vec.end());
  TIME_OPERATION_NONCT("MakePackedPlaintext", op.getPlaintext(),
                       cc->MakePackedPlaintext(vec64), plaintexts);
}

void Interpreter::visit(MakeCKKSPackedPlaintextOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto valType = op.getValue().getType();
  auto elemType = cast<RankedTensorType>(valType).getElementType();

  if (elemType.isF32()) {
    const auto& vec = *floatVectors.at(op.getValue());
    std::vector<double> vecDouble(vec.begin(), vec.end());
    TIME_OPERATION_NONCT("MakeCKKSPackedPlaintext", op.getPlaintext(),
                         cc->MakeCKKSPackedPlaintext(vecDouble), plaintexts);
  } else if (elemType.isF64()) {
    const auto& vec = *doubleVectors.at(op.getValue());
    TIME_OPERATION_NONCT("MakeCKKSPackedPlaintext", op.getPlaintext(),
                         cc->MakeCKKSPackedPlaintext(vec), plaintexts);
  } else if (elemType.isInteger()) {
    const auto& vec = *intVectors.at(op.getValue());
    std::vector<double> vecDouble(vec.begin(), vec.end());
    TIME_OPERATION_NONCT("MakeCKKSPackedPlaintext", op.getPlaintext(),
                         cc->MakeCKKSPackedPlaintext(vecDouble), plaintexts);
  }
}

void Interpreter::visit(GenParamsOp op) {
  int64_t mulDepth = op.getMulDepthAttr().getValue().getSExtValue();
  int64_t plainMod = op.getPlainModAttr().getValue().getSExtValue();
  int64_t evalAddCount = op.getEvalAddCountAttr().getValue().getSExtValue();
  int64_t keySwitchCount = op.getKeySwitchCountAttr().getValue().getSExtValue();

  auto params = std::make_shared<CCParamsT>();
  params->SetMultiplicativeDepth(mulDepth);
  if (plainMod > 0) params->SetPlaintextModulus(plainMod);
  if (op.getRingDim() != 0) params->SetRingDim(op.getRingDim());
  if (op.getBatchSize() != 0) params->SetBatchSize(op.getBatchSize());
  if (op.getFirstModSize() != 0) params->SetFirstModSize(op.getFirstModSize());
  if (op.getScalingModSize() != 0)
    params->SetScalingModSize(op.getScalingModSize());
  if (evalAddCount > 0) params->SetEvalAddCount(evalAddCount);
  if (keySwitchCount > 0) params->SetKeySwitchCount(keySwitchCount);
  if (op.getDigitSize() != 0) params->SetDigitSize(op.getDigitSize());
  if (op.getNumLargeDigits() != 0)
    params->SetNumLargeDigits(op.getNumLargeDigits());
  if (op.getMaxRelinSkDeg() != 0)
    params->SetMaxRelinSkDeg(op.getMaxRelinSkDeg());
  if (op.getInsecure()) params->SetSecurityLevel(HEStd_NotSet);
  if (op.getEncryptionTechniqueExtended())
    params->SetEncryptionTechnique(EXTENDED);
  if (!op.getKeySwitchingTechniqueBV())
    params->SetKeySwitchTechnique(HYBRID);
  else
    params->SetKeySwitchTechnique(BV);
  if (op.getScalingTechniqueFixedManual())
    params->SetScalingTechnique(FIXEDMANUAL);

  params_.insert_or_assign(op.getResult(), std::move(params));
}

void Interpreter::visit(GenContextOp op) {
  auto params = params_.at(op.getParams());
  CryptoContextT cc;
  TIME_OPERATION_VOID("GenContext", cc = GenCryptoContext(*params));
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  if (op.getSupportFHE()) {
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
  }
  cryptoContexts[op.getResult()] = cc;
}

void Interpreter::visit(GenRotKeyOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto pk = privateKeys.at(op.getPrivateKey());
  std::vector<int32_t> rotIndices(op.getIndices().begin(),
                                  op.getIndices().end());
  for (auto index : rotIndices) {
    std::cout << "Generating rot key for " << index << "\n";
    TIME_OPERATION_VOID("GenRotKey", cc->EvalRotateKeyGen(pk, {index}));
  }
}

void Interpreter::visit(GenMulKeyOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto pk = privateKeys.at(op.getPrivateKey());
  TIME_OPERATION_VOID("GenMulKey", cc->EvalMultKeyGen(pk));
}

void Interpreter::visit(GenBootstrapKeyOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto pk = privateKeys.at(op.getPrivateKey());
  // Use full packing - ring dimension / 2
  auto numSlots = cc->GetRingDimension() / 2;
  TIME_OPERATION_VOID("GenBootstrapKey", cc->EvalBootstrapKeyGen(pk, numSlots));
}

void Interpreter::visit(SetupBootstrapOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  std::vector<uint32_t> levelBudget = {
      static_cast<uint32_t>(
          op.getLevelBudgetEncode().getValue().getSExtValue()),
      static_cast<uint32_t>(
          op.getLevelBudgetDecode().getValue().getSExtValue())};
  TIME_OPERATION_VOID("SetupBootstrap", cc->EvalBootstrapSetup(levelBudget));
}

void Interpreter::visit(FastRotationOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getInput());
  auto index = op.getIndex().getZExtValue();
  auto digits = fastRotPrecomps.at(op.getPrecomputedDigitDecomp());
  auto m = 2 * cc->GetRingDimension();
  TIME_OPERATION("FastRotation", op.getResult(),
                 cc->EvalFastRotation(ct, index, m, digits));
}

void Interpreter::visit(FastRotationPrecomputeOp op) {
  auto cc = cryptoContexts.at(op.getCryptoContext());
  auto ct = ciphertexts.at(op.getInput());
  TIME_OPERATION_NONCT("FastRotationPrecompute", op.getResult(),
                       cc->EvalFastRotationPrecompute(ct), fastRotPrecomps);
}

void Interpreter::visit(lwe::RLWEDecodeOp op) {
  auto plaintext = plaintexts.at(op.getInput());
  bool isCKKS = llvm::isa<lwe::InverseCanonicalEncodingAttr>(op.getEncoding());

  if (auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType())) {
    auto shape = tensorTy.getShape();
    auto nonUnitDims = llvm::count_if(shape, [](auto dim) { return dim != 1; });
    if (nonUnitDims != 1) {
      op->emitError()
          << "Only 1D tensors with one non-unit dimension supported";
      return;
    }

    int64_t size = 0;
    for (auto dim : shape) {
      if (dim != 1) {
        size = dim;
        break;
      }
    }
    plaintext->SetLength(size);

    if (isCKKS) {
      auto ckksValues = plaintext->GetCKKSPackedValue();
      auto elemType = tensorTy.getElementType();

      if (elemType.isF64()) {
        auto result = std::make_shared<std::vector<double>>();
        result->reserve(ckksValues.size());
        for (const auto& val : ckksValues) {
          result->push_back(val.real());
        }
        doubleVectors[op.getResult()] = result;
      } else {
        auto result = std::make_shared<std::vector<float>>();
        result->reserve(ckksValues.size());
        for (const auto& val : ckksValues) {
          result->push_back(static_cast<float>(val.real()));
        }
        floatVectors[op.getResult()] = result;
      }
    } else {
      auto packedValues = plaintext->GetPackedValue();
      auto result = std::make_shared<std::vector<int>>();
      result->reserve(packedValues.size());
      for (const auto& val : packedValues) {
        result->push_back(static_cast<int>(val));
      }
      intVectors[op.getResult()] = result;
    }
  } else {
    // Scalar result
    if (isCKKS) {
      auto ckksValues = plaintext->GetCKKSPackedValue();
      auto elemType = op.getResult().getType();

      if (elemType.isF64()) {
        doubleValues[op.getResult()] = ckksValues[0].real();
      } else {
        floatValues[op.getResult()] = static_cast<float>(ckksValues[0].real());
      }
    } else {
      auto packedValues = plaintext->GetPackedValue();
      intValues[op.getResult()] = static_cast<int>(packedValues[0]);
    }
  }
}

void initContext(MLIRContext& context) {
  mlir::DialectRegistry registry;
  registry.insert<OpenfheDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<mod_arith::ModArithDialect>();
  registry.insert<polynomial::PolynomialDialect>();
  registry.insert<rns::RNSDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<affine::AffineDialect>();
  rns::registerExternalRNSTypeInterfaces(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

OwningOpRef<ModuleOp> parse(MLIRContext* context, const std::string& mlirStr) {
  return parseSourceString<ModuleOp>(mlirStr, context);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
