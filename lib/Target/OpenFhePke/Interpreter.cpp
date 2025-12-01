#include "lib/Target/OpenFhePke/Interpreter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
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
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project

#ifdef OPENFHE_ENABLE_TIMING
#define TIME_OPERATION_VOID(op_name, code)                  \
  do {                                                      \
    auto start = std::chrono::high_resolution_clock::now(); \
    code;                                                   \
    auto end = std::chrono::high_resolution_clock::now();   \
    timingResults[op_name].totalTime += (end - start);      \
    timingResults[op_name].count++;                         \
  } while (0)
#define TIME_OPERATION(op_name, result_val, code)           \
  do {                                                      \
    auto start = std::chrono::high_resolution_clock::now(); \
    auto result = code;                                     \
    auto end = std::chrono::high_resolution_clock::now();   \
    timingResults[op_name].totalTime += (end - start);      \
    timingResults[op_name].count++;                         \
    env.insert_or_assign(result_val, std::move(result));    \
  } while (0)
#else
#define TIME_OPERATION_VOID(op_name, code) \
  do {                                     \
    code;                                  \
  } while (0)
#define TIME_OPERATION(op_name, result_val, code) \
  do {                                            \
    env.insert_or_assign(result_val, code);       \
  } while (0)
#endif

#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"       // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"      // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
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
#include "src/pke/include/scheme/ckksrns/gen-cryptocontext-ckksrns-params.h"  // from @openfhe
#include "src/pke/include/scheme/ckksrns/gen-cryptocontext-ckksrns.h"  // from @openfhe

namespace mlir {
namespace heir {
namespace openfhe {

namespace {

TypedCppValue getValueFromDenseElementsAttr(DenseElementsAttr elementsAttr) {
  // We flatten all tensors to a 1D shape in row-major order.
  if (elementsAttr.getType().getElementType().isInteger(32) ||
      elementsAttr.getType().getElementType().isInteger(64)) {
    std::vector<int> values;
    for (auto val : elementsAttr.getValues<APInt>()) {
      values.push_back(static_cast<int>(val.getSExtValue()));
    }
    return values;
  } else if (elementsAttr.getType().getElementType().isF32()) {
    // Use float for F32
    std::vector<float> values;
    for (auto val : elementsAttr.getValues<APFloat>()) {
      values.push_back(static_cast<float>(val.convertToFloat()));
    }
    return values;
  } else if (elementsAttr.getType().getElementType().isF64()) {
    // Use double for F64
    std::vector<double> values;
    for (auto val : elementsAttr.getValues<APFloat>()) {
      values.push_back(val.convertToDouble());
    }
    return values;
  }
  return TypedCppValue();
}

}  // namespace

using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContextCKKSRNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;
using FastRotPrecompT = std::shared_ptr<std::vector<DCRTPoly>>;

std::vector<TypedCppValue> Interpreter::interpret(
    const std::string& entryFunction, ArrayRef<TypedCppValue> inputValues) {
  env.clear();
  func::FuncOp func = module.lookupSymbol<func::FuncOp>(entryFunction);
  std::vector<TypedCppValue> results;

  FunctionType funcType = func.getFunctionType();
  SmallVector<Type> argTypes(funcType.getInputs());
  SmallVector<Type> returnTypes(funcType.getResults());

  if (argTypes.size() != inputValues.size()) {
    func->emitError() << "Input size does not match function signature";
  }

  for (const auto& [argIndex, argTy] : llvm::enumerate(argTypes)) {
    env.insert({func.getBody().getArgument(argIndex), inputValues[argIndex]});
  }

  llvm::outs() << "Interpreting function: " << func.getName()
               << " with signature " << funcType << " and "
               << inputValues.size() << " interpreted arguments\n";

  // Walk only the operations in the entry block, not nested regions
  // Nested regions (like loop bodies) will be handled by their parent ops
  for (auto& op : func.getBody().front().getOperations()) {
    if (auto returnOp = dyn_cast<func::ReturnOp>(&op)) {
      for (auto retVal : returnOp.getOperands()) {
        results.push_back(env.at(retVal));
      }
      llvm::outs() << "Function returned " << results.size() << " values\n";
    } else {
      visit(&op);
    }
  }

  return results;
}

void Interpreter::visit(Operation* op) {
  llvm::TypeSwitch<Operation*>(op)
      .Case<arith::ConstantOp, arith::AddIOp, arith::AddFOp, arith::SubIOp,
            arith::MulIOp, arith::MulFOp, arith::DivSIOp, arith::RemSIOp,
            arith::AndIOp, arith::CmpIOp, arith::SelectOp, arith::ExtUIOp,
            arith::ExtSIOp, arith::ExtFOp, arith::FloorDivSIOp,
            arith::IndexCastOp, arith::MinSIOp, arith::MaxSIOp, tensor::EmptyOp,
            tensor::ExtractOp, tensor::InsertOp, tensor::SplatOp,
            tensor::FromElementsOp, tensor::ConcatOp, tensor::ExtractSliceOp,
            tensor::InsertSliceOp, tensor::CollapseShapeOp,
            tensor::ExpandShapeOp, linalg::BroadcastOp, scf::ForOp, scf::IfOp,
            scf::YieldOp, affine::AffineForOp, affine::AffineYieldOp,
            lwe::RLWEDecodeOp, AddOp, AddPlainOp, SubOp, SubPlainOp, MulOp,
            MulNoRelinOp, MulPlainOp, MulConstOp, NegateOp, SquareOp, RelinOp,
            ModReduceOp, LevelReduceOp, RotOp, AutomorphOp, KeySwitchOp,
            BootstrapOp, EncryptOp, DecryptOp, MakePackedPlaintextOp,
            MakeCKKSPackedPlaintextOp, GenParamsOp, GenContextOp, GenRotKeyOp,
            GenMulKeyOp, GenBootstrapKeyOp, SetupBootstrapOp, FastRotationOp,
            FastRotationPrecomputeOp>([&](auto op) { visit(op); })
      .Default([&](Operation* op) {
        OperationName opName = op->getName();
        op->emitError() << "Unsupported operation " << opName.getStringRef()
                        << " in interpreter";
      });
}

void Interpreter::visit(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    env.insert_or_assign(op.getResult(), static_cast<int>(intAttr.getInt()));
  } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    // Use float for 32-bit and smaller, double for 64-bit and larger
    if (floatAttr.getType().isF32()) {
      env.insert_or_assign(op.getResult(),
                           static_cast<float>(floatAttr.getValueAsDouble()));
    } else {
      // F64 and larger types use double
      env.insert_or_assign(op.getResult(), floatAttr.getValueAsDouble());
    }
  } else if (auto elementsAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    auto value = getValueFromDenseElementsAttr(elementsAttr);
    if (std::holds_alternative<std::monostate>(value.value)) {
      op->emitError() << "Unsupported DenseElementsAttr type "
                      << elementsAttr.getType().getElementType();
    }
    env.insert_or_assign(op.getResult(), std::move(value));
  } else if (auto denseResourceAttr =
                 dyn_cast<DenseResourceElementsAttr>(valueAttr)) {
    const auto data = denseResourceAttr.getData();
    auto denseElementsAttr =
        DenseElementsAttr::getFromRawBuffer(denseResourceAttr.getType(), data);
    auto value = getValueFromDenseElementsAttr(denseElementsAttr);
    if (std::holds_alternative<std::monostate>(value.value)) {
      op->emitError() << "Unsupported DenseResourceElementsAttr type "
                      << denseResourceAttr.getType().getElementType();
    }
    env.insert_or_assign(op.getResult(), std::move(value));
  } else {
    op->emitError() << "Unsupported constant attribute type " << valueAttr;
  }
}

TypedCppValue Interpreter::applyBinop(
    Operation* op, const TypedCppValue& lhs, const TypedCppValue& rhs,
    std::function<int(int, int)> intFunc,
    std::function<double(double, double)> doubleFunc) {
  if (std::holds_alternative<int>(lhs.value) &&
      std::holds_alternative<int>(rhs.value)) {
    return TypedCppValue(
        intFunc(std::get<int>(lhs.value), std::get<int>(rhs.value)));
  } else if (std::holds_alternative<float>(lhs.value) &&
             std::holds_alternative<float>(rhs.value)) {
    return TypedCppValue(static_cast<float>(
        doubleFunc(std::get<float>(lhs.value), std::get<float>(rhs.value))));
  } else if (std::holds_alternative<double>(lhs.value) &&
             std::holds_alternative<double>(rhs.value)) {
    return TypedCppValue(
        doubleFunc(std::get<double>(lhs.value), std::get<double>(rhs.value)));
  } else if (std::holds_alternative<std::vector<int>>(lhs.value) &&
             std::holds_alternative<std::vector<int>>(rhs.value)) {
    const auto& lhsVec = std::get<std::vector<int>>(lhs.value);
    const auto& rhsVec = std::get<std::vector<int>>(rhs.value);
    if (lhsVec.size() != rhsVec.size()) {
      op->emitError("Vector size mismatch in binary operation");
      return TypedCppValue();
    }
    std::vector<int> resultVec(lhsVec.size());
    for (size_t i = 0; i < lhsVec.size(); ++i)
      resultVec[i] = intFunc(lhsVec[i], rhsVec[i]);
    return TypedCppValue(resultVec);
  } else if (std::holds_alternative<std::vector<float>>(lhs.value) &&
             std::holds_alternative<std::vector<float>>(rhs.value)) {
    const auto& lhsVec = std::get<std::vector<float>>(lhs.value);
    const auto& rhsVec = std::get<std::vector<float>>(rhs.value);
    if (lhsVec.size() != rhsVec.size()) {
      op->emitError("Vector size mismatch in binary operation");
      return TypedCppValue();
    }
    std::vector<float> resultVec(lhsVec.size());
    for (size_t i = 0; i < lhsVec.size(); ++i)
      resultVec[i] = static_cast<float>(doubleFunc(lhsVec[i], rhsVec[i]));
    return TypedCppValue(resultVec);
  } else if (std::holds_alternative<std::vector<double>>(lhs.value) &&
             std::holds_alternative<std::vector<double>>(rhs.value)) {
    const auto& lhsVec = std::get<std::vector<double>>(lhs.value);
    const auto& rhsVec = std::get<std::vector<double>>(rhs.value);
    if (lhsVec.size() != rhsVec.size()) {
      op->emitError("Vector size mismatch in binary operation");
      return TypedCppValue();
    }
    std::vector<double> resultVec(lhsVec.size());
    for (size_t i = 0; i < lhsVec.size(); ++i)
      resultVec[i] = doubleFunc(lhsVec[i], rhsVec[i]);
    return TypedCppValue(resultVec);
  } else {
    op->emitError() << "Type mismatch in binary operation; " << op
                    << " with interpreted operand type variants "
                    << lhs.value.index() << " and " << rhs.value.index();
    return TypedCppValue();
  }
}

void Interpreter::visit(arith::AddIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return a + b; },
      [](float a, float b) { return a + b; });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::AddFOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return a + b; },
      [](float a, float b) { return a + b; });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::SubIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return a - b; },
      [](float a, float b) { return a - b; });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::MulIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return a * b; },
      [](float a, float b) { return a * b; });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::MulFOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return a * b; },
      [](float a, float b) { return a * b; });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::DivSIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return a / b; },
      [](float a, float b) { return a / b; });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::RemSIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return a % b; },
      [](float a, float b) { return std::fmod(a, b); });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::AndIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return a && b; },
      [](float a, float b) { return static_cast<float>(a && b); });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::MaxSIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return std::max(a, b); },
      [](float a, float b) { return std::max(a, b); });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::MinSIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs, [](int a, int b) { return std::min(a, b); },
      [](float a, float b) { return std::min(a, b); });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::CmpIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());

  auto cmpFunc = [&](auto a, auto b) -> bool {
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:
        return a == b;
      case arith::CmpIPredicate::ne:
        return a != b;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        return a < b;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        return a <= b;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        return a > b;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        return a >= b;
    }
    op.emitError("Unknown predicate");
    return false;
  };

  auto result = applyBinop(
      op, lhs, rhs, [&](int a, int b) { return cmpFunc(a, b) ? 1 : 0; },
      [&](float a, float b) { return cmpFunc(a, b) ? 1.0f : 0.0f; });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::SelectOp op) {
  auto condition = env.at(op.getCondition());
  auto trueValue = env.at(op.getTrueValue());
  auto falseValue = env.at(op.getFalseValue());

  bool condBool = false;
  if (std::holds_alternative<bool>(condition.value)) {
    condBool = std::get<bool>(condition.value);
  } else if (std::holds_alternative<int>(condition.value)) {
    condBool = std::get<int>(condition.value) != 0;
  } else if (std::holds_alternative<float>(condition.value)) {
    condBool = std::get<float>(condition.value) != 0.0f;
  } else if (std::holds_alternative<double>(condition.value)) {
    condBool = std::get<double>(condition.value) != 0.0;
  } else {
    op.emitError("Select condition must be bool, int, float, or double");
  }

  env.insert_or_assign(op.getResult(),
                       condBool ? std::move(trueValue) : std::move(falseValue));
}

void Interpreter::visit(arith::ExtUIOp op) {
  auto operand = env.at(op.getIn());
  // For unsigned extension, we just convert the value to a larger type
  if (std::holds_alternative<int>(operand.value)) {
    env.insert_or_assign(op.getResult(), std::get<int>(operand.value));
  } else {
    op.emitError("ExtUIOp requires int operand");
  }
}

void Interpreter::visit(arith::FloorDivSIOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto result = applyBinop(
      op, lhs, rhs,
      [](int a, int b) { return std::floor(static_cast<float>(a) / b); },
      [](float a, float b) {
        llvm_unreachable("unexpceted float operands in FloorDivSIOp");
        return std::floor(a / b);
      });
  env.insert_or_assign(op.getResult(), std::move(result));
}

void Interpreter::visit(arith::ExtSIOp op) {
  auto operand = env.at(op.getIn());
  // For signed extension, we just convert the value to a larger type
  if (std::holds_alternative<int>(operand.value)) {
    env.insert_or_assign(op.getResult(), std::get<int>(operand.value));
  } else {
    op.emitError("ExtSIOp requires int operand");
  }
}

void Interpreter::visit(arith::ExtFOp op) {
  auto operand = env.at(op.getIn());
  auto outType = op.getOut().getType();

  // Determine output precision based on output type
  auto floatType = cast<FloatType>(getElementTypeOrSelf(outType));
  bool outputIsDouble = floatType.getIntOrFloatBitWidth() > 32;

  // ExtFOp can extend float precision or convert int to float/double
  if (std::holds_alternative<float>(operand.value)) {
    if (outputIsDouble) {
      env.insert_or_assign(op.getResult(),
                           static_cast<double>(std::get<float>(operand.value)));
    } else {
      env.insert_or_assign(op.getResult(), std::get<float>(operand.value));
    }
  } else if (std::holds_alternative<double>(operand.value)) {
    env.insert_or_assign(op.getResult(), std::get<double>(operand.value));
  } else if (std::holds_alternative<std::vector<float>>(operand.value)) {
    const auto& vec = std::get<std::vector<float>>(operand.value);
    if (outputIsDouble) {
      std::vector<double> result(vec.begin(), vec.end());
      env.insert_or_assign(op.getResult(), std::move(result));
    } else {
      env.insert_or_assign(op.getResult(), vec);
    }
  } else if (std::holds_alternative<std::vector<double>>(operand.value)) {
    env.insert_or_assign(op.getResult(),
                         std::get<std::vector<double>>(operand.value));
  } else {
    op->emitError() << "ExtFOp received operand with variant index "
                    << operand.value.index();
  }
}

void Interpreter::visit(arith::IndexCastOp op) {
  auto operand = env.at(op.getIn());
  // IndexCastOp converts between index and integer types
  if (std::holds_alternative<int>(operand.value)) {
    env.insert_or_assign(op.getResult(), std::get<int>(operand.value));
  } else {
    op.emitError("IndexCastOp requires int operand");
  }
}

int Interpreter::getFlattenedTensorIndex(Value tensor, ValueRange indices) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto shape = tensorType.getShape();
  int accum = std::get<int>(env.at(indices[0]).value);
  for (size_t i = 1; i < shape.size(); ++i) {
    accum = accum * shape[i] + std::get<int>(env.at(indices[i]).value);
  }
  return accum;
}

void Interpreter::visit(tensor::EmptyOp op) {
  // Create an empty vector of the appropriate size and type
  auto tensorType = op.getResult().getType();
  auto numElements = tensorType.getNumElements();
  auto elementType = tensorType.getElementType();

  if (elementType.isInteger(32) || elementType.isInteger(64)) {
    env.insert_or_assign(op.getResult(), std::vector<int>(numElements));
  } else if (elementType.isF32()) {
    env.insert_or_assign(op.getResult(), std::vector<float>(numElements));
  } else if (elementType.isF64()) {
    env.insert_or_assign(op.getResult(), std::vector<double>(numElements));
  } else if (isa<lwe::LWEPlaintextType>(elementType)) {
    env.insert_or_assign(op.getResult(), std::vector<PlaintextT>(numElements));
  } else {
    env.insert_or_assign(op.getResult(), std::vector<CiphertextT>(numElements));
  }
}

void Interpreter::visit(tensor::ExtractOp op) {
  auto tensor = env.at(op.getTensor());
  int index = getFlattenedTensorIndex(op.getTensor(), op.getIndices());
  if (std::holds_alternative<std::vector<int>>(tensor.value)) {
    auto& vec = std::get<std::vector<int>>(tensor.value);
    env.insert_or_assign(op.getResult(), vec[index]);
  } else if (std::holds_alternative<std::vector<float>>(tensor.value)) {
    auto& vec = std::get<std::vector<float>>(tensor.value);
    env.insert_or_assign(op.getResult(), vec[index]);
  } else if (std::holds_alternative<std::vector<double>>(tensor.value)) {
    auto& vec = std::get<std::vector<double>>(tensor.value);
    env.insert_or_assign(op.getResult(), vec[index]);
  } else if (std::holds_alternative<std::vector<PlaintextT>>(tensor.value)) {
    auto& vec = std::get<std::vector<PlaintextT>>(tensor.value);
    env.insert_or_assign(op.getResult(), vec[index]);
  } else if (std::holds_alternative<std::vector<CiphertextT>>(tensor.value)) {
    auto& vec = std::get<std::vector<CiphertextT>>(tensor.value);
    env.insert_or_assign(op.getResult(), vec[index]);
  } else {
    op.emitError("Unsupported tensor type in ExtractOp");
  }
}

void Interpreter::visit(tensor::InsertOp op) {
  auto scalar = env.at(op.getScalar());
  auto dest = env.at(op.getDest());
  auto indices = op.getIndices();

  // Calculate the flattened index for multi-dimensional tensors
  int index = getFlattenedTensorIndex(op.getDest(), indices);

  // Make a copy of the destination tensor and insert the scalar
  if (std::holds_alternative<std::vector<int>>(dest.value)) {
    auto vec = std::get<std::vector<int>>(dest.value);
    vec[index] = std::get<int>(scalar.value);
    env.insert_or_assign(op.getResult(), std::move(vec));
  } else if (std::holds_alternative<std::vector<float>>(dest.value)) {
    auto vec = std::get<std::vector<float>>(dest.value);
    vec[index] = std::get<float>(scalar.value);
    env.insert_or_assign(op.getResult(), std::move(vec));
  } else if (std::holds_alternative<std::vector<double>>(dest.value)) {
    auto vec = std::get<std::vector<double>>(dest.value);
    vec[index] = std::get<double>(scalar.value);
    env.insert_or_assign(op.getResult(), std::move(vec));
  } else if (std::holds_alternative<std::vector<CiphertextT>>(dest.value)) {
    auto vec = std::get<std::vector<CiphertextT>>(dest.value);
    vec[index] = std::get<CiphertextT>(scalar.value);
    env.insert_or_assign(op.getResult(), std::move(vec));
  } else if (std::holds_alternative<std::vector<PlaintextT>>(dest.value)) {
    auto vec = std::get<std::vector<PlaintextT>>(dest.value);
    vec[index] = std::get<PlaintextT>(scalar.value);
    env.insert_or_assign(op.getResult(), std::move(vec));
  } else {
    op.emitError("Unsupported tensor type in InsertOp");
  }
}

void Interpreter::visit(tensor::SplatOp op) {
  auto input = env.at(op.getInput());
  auto tensorType = op.getResult().getType();
  auto numElements = tensorType.getNumElements();

  if (std::holds_alternative<int>(input.value)) {
    int val = std::get<int>(input.value);
    env.insert_or_assign(op.getResult(), std::vector<int>(numElements, val));
  } else if (std::holds_alternative<float>(input.value)) {
    float val = std::get<float>(input.value);
    env.insert_or_assign(op.getResult(), std::vector<float>(numElements, val));
  } else if (std::holds_alternative<double>(input.value)) {
    double val = std::get<double>(input.value);
    env.insert_or_assign(op.getResult(), std::vector<double>(numElements, val));
  } else {
    op.emitError("Unsupported input type in SplatOp");
  }
}

void Interpreter::visit(tensor::FromElementsOp op) {
  auto elements = op.getElements();
  if (elements.empty()) {
    op.emitError("FromElementsOp requires at least one element");
  }

  auto firstElement = env.at(elements[0]);
  if (std::holds_alternative<int>(firstElement.value)) {
    std::vector<int> result;
    result.reserve(elements.size());
    for (auto element : elements) {
      result.push_back(std::get<int>(env.at(element).value));
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<float>(firstElement.value)) {
    std::vector<float> result;
    result.reserve(elements.size());
    for (auto element : elements) {
      result.push_back(std::get<float>(env.at(element).value));
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<double>(firstElement.value)) {
    std::vector<double> result;
    result.reserve(elements.size());
    for (auto element : elements) {
      result.push_back(std::get<double>(env.at(element).value));
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<CiphertextT>(firstElement.value)) {
    std::vector<CiphertextT> result;
    result.reserve(elements.size());
    for (auto element : elements) {
      result.push_back(std::get<CiphertextT>(env.at(element).value));
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<PlaintextT>(firstElement.value)) {
    std::vector<PlaintextT> result;
    result.reserve(elements.size());
    for (auto element : elements) {
      result.push_back(std::get<PlaintextT>(env.at(element).value));
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else {
    op->emitError() << "FromElementsOp received element with variant index "
                    << firstElement.value.index();
  }
}

void Interpreter::visit(tensor::ConcatOp op) {
  auto inputs = op.getInputs();
  auto firstInput = env.at(inputs[0]);

  if (std::holds_alternative<std::vector<int>>(firstInput.value)) {
    std::vector<int> result;
    for (auto input : inputs) {
      const auto& vec = std::get<std::vector<int>>(env.at(input).value);
      result.insert(result.end(), vec.begin(), vec.end());
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<float>>(firstInput.value)) {
    std::vector<float> result;
    for (auto input : inputs) {
      const auto& vec = std::get<std::vector<float>>(env.at(input).value);
      result.insert(result.end(), vec.begin(), vec.end());
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<double>>(firstInput.value)) {
    std::vector<double> result;
    for (auto input : inputs) {
      const auto& vec = std::get<std::vector<double>>(env.at(input).value);
      result.insert(result.end(), vec.begin(), vec.end());
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<CiphertextT>>(
                 firstInput.value)) {
    std::vector<CiphertextT> result;
    for (auto input : inputs) {
      const auto& vec = std::get<std::vector<CiphertextT>>(env.at(input).value);
      result.insert(result.end(), vec.begin(), vec.end());
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<PlaintextT>>(
                 firstInput.value)) {
    std::vector<PlaintextT> result;
    for (auto input : inputs) {
      const auto& vec = std::get<std::vector<PlaintextT>>(env.at(input).value);
      result.insert(result.end(), vec.begin(), vec.end());
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else {
    op.emitError("Unsupported tensor type in ConcatOp");
  }
}

void Interpreter::visit(tensor::ExtractSliceOp op) {
  auto source = env.at(op.getSource());
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

  if (std::holds_alternative<std::vector<int>>(source.value)) {
    const auto& vec = std::get<std::vector<int>>(source.value);
    std::vector<int> result;
    result.reserve(totalElements);
    for (int64_t i = 0; i < totalElements; ++i) {
      result.push_back(vec[extractElement(i)]);
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<float>>(source.value)) {
    const auto& vec = std::get<std::vector<float>>(source.value);
    std::vector<float> result;
    result.reserve(totalElements);
    for (int64_t i = 0; i < totalElements; ++i) {
      result.push_back(vec[extractElement(i)]);
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<double>>(source.value)) {
    const auto& vec = std::get<std::vector<double>>(source.value);
    std::vector<double> result;
    result.reserve(totalElements);
    for (int64_t i = 0; i < totalElements; ++i) {
      result.push_back(vec[extractElement(i)]);
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<CiphertextT>>(source.value)) {
    const auto& vec = std::get<std::vector<CiphertextT>>(source.value);
    std::vector<CiphertextT> result;
    result.reserve(totalElements);
    for (int64_t i = 0; i < totalElements; ++i) {
      result.push_back(vec[extractElement(i)]);
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<PlaintextT>>(source.value)) {
    const auto& vec = std::get<std::vector<PlaintextT>>(source.value);
    std::vector<PlaintextT> result;
    result.reserve(totalElements);
    for (int64_t i = 0; i < totalElements; ++i) {
      result.push_back(vec[extractElement(i)]);
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else {
    op.emitError("Unsupported tensor type in ExtractSliceOp");
  }
}

void Interpreter::visit(tensor::InsertSliceOp op) {
  auto source = env.at(op.getSource());
  auto dest = env.at(op.getDest());
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

  if (std::holds_alternative<std::vector<int>>(dest.value)) {
    auto result = std::get<std::vector<int>>(dest.value);
    const auto& srcVec = std::get<std::vector<int>>(source.value);
    for (int64_t i = 0; i < totalElements; ++i) {
      result[insertElement(i)] = srcVec[i];
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<float>>(dest.value)) {
    auto result = std::get<std::vector<float>>(dest.value);
    const auto& srcVec = std::get<std::vector<float>>(source.value);
    for (int64_t i = 0; i < totalElements; ++i) {
      result[insertElement(i)] = srcVec[i];
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<double>>(dest.value)) {
    auto result = std::get<std::vector<double>>(dest.value);
    const auto& srcVec = std::get<std::vector<double>>(source.value);
    for (int64_t i = 0; i < totalElements; ++i) {
      result[insertElement(i)] = srcVec[i];
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<CiphertextT>>(dest.value)) {
    auto result = std::get<std::vector<CiphertextT>>(dest.value);
    const auto& srcVec = std::get<std::vector<CiphertextT>>(source.value);
    for (int64_t i = 0; i < totalElements; ++i) {
      result[insertElement(i)] = srcVec[i];
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else if (std::holds_alternative<std::vector<PlaintextT>>(dest.value)) {
    auto result = std::get<std::vector<PlaintextT>>(dest.value);
    const auto& srcVec = std::get<std::vector<PlaintextT>>(source.value);
    for (int64_t i = 0; i < totalElements; ++i) {
      result[insertElement(i)] = srcVec[i];
    }
    env.insert_or_assign(op.getResult(), std::move(result));
  } else {
    op.emitError("Unsupported tensor type in InsertSliceOp");
  }
}

void Interpreter::visit(tensor::CollapseShapeOp op) {
  auto src = env.at(op.getSrc());
  // CollapseShape just changes the shape metadata, the underlying flat data
  // remains the same Since we already flatten all tensors to 1D, we can just
  // copy the value
  env.insert_or_assign(op.getResult(), src);
}

void Interpreter::visit(tensor::ExpandShapeOp op) {
  auto src = env.at(op.getSrc());
  // ExpandShape just changes the shape metadata, the underlying flat data
  // remains the same Since we already flatten all tensors to 1D, we can just
  // copy the value
  env.insert_or_assign(op.getResult(), src);
}

void Interpreter::visit(linalg::BroadcastOp op) {
  // BroadcastOp copies the input op into a new tensor by adding the specified
  // dims.
  auto input = env.at(op.getInput());
  auto resultType = cast<RankedTensorType>(op->getResults()[0].getType());
  auto resultShape = resultType.getShape();
  auto inputType = cast<RankedTensorType>(op.getInput().getType());
  auto inputShape = inputType.getShape();
  auto numOutputElements = resultType.getNumElements();

  auto calculate = [&](const auto& inputVec, auto& outputVec) {
    outputVec.resize(numOutputElements);

    // Create a map of the broadcast dimensions.
    DenseMap<int64_t, bool> broadcastDims;
    for (const auto& dim : op.getDimensions()) {
      broadcastDims[dim] = true;
    }

    // Iterate over the output tensor's elements
    for (int64_t i = 0; i < numOutputElements; ++i) {
      // Calculate the multi-dimensional index in the output tensor by
      // unflattening the index in the output shape.
      std::vector<int64_t> outputIndices(resultShape.size());
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
      std::vector<int64_t> inputIndices(inputShape.size());
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

  Value result = op->getResults()[0];
  if (std::holds_alternative<std::vector<int>>(input.value)) {
    const auto& inputVec = std::get<std::vector<int>>(input.value);
    std::vector<int> resultVec;
    calculate(inputVec, resultVec);
    env.insert_or_assign(result, std::move(resultVec));
  } else if (std::holds_alternative<std::vector<float>>(input.value)) {
    const auto& inputVec = std::get<std::vector<float>>(input.value);
    std::vector<float> resultVec;
    calculate(inputVec, resultVec);
    env.insert_or_assign(result, std::move(resultVec));
  } else if (std::holds_alternative<std::vector<double>>(input.value)) {
    const auto& inputVec = std::get<std::vector<double>>(input.value);
    std::vector<double> resultVec;
    calculate(inputVec, resultVec);
    env.insert_or_assign(result, std::move(resultVec));
  } else {
    op.emitError("Unsupported tensor type in BroadcastOp");
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
  auto lowerBound = std::get<int>(env.at(op.getLowerBound()).value);
  auto upperBound = std::get<int>(env.at(op.getUpperBound()).value);
  auto step = std::get<int>(env.at(op.getStep()).value);

  // Initialize iter args with initial values
  std::vector<TypedCppValue> iterArgs;
  for (auto initArg : op.getInitArgs()) {
    iterArgs.push_back(env.at(initArg));
  }

  // Execute the loop
  for (int i = lowerBound; i < upperBound; i += step) {
    // Set up induction variable and iter args in env
    env.erase(op.getInductionVar());
    env.insert({op.getInductionVar(), TypedCppValue(i)});

    for (auto [blockArg, iterArg] :
         llvm::zip(op.getRegionIterArgs(), iterArgs)) {
      env.erase(blockArg);
      env.insert({blockArg, iterArg});
    }

    // Walk the body and execute operations
    std::vector<TypedCppValue> yieldResults;
    for (auto& bodyOp : op.getBody()->getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
        // Capture the yielded values
        for (auto yieldOperand : yieldOp.getOperands()) {
          yieldResults.push_back(env.at(yieldOperand));
        }
      } else {
        visit(&bodyOp);
      }
    }

    // Update iter args for next iteration
    iterArgs = std::move(yieldResults);
  }

  // Store final results
  for (auto [result, finalValue] : llvm::zip(op.getResults(), iterArgs)) {
    env.insert_or_assign(result, finalValue);
  }
}

void Interpreter::visit(scf::IfOp op) {
  auto condition = env.at(op.getCondition());

  bool condBool = false;
  if (std::holds_alternative<bool>(condition.value)) {
    condBool = std::get<bool>(condition.value);
  } else if (std::holds_alternative<int>(condition.value)) {
    condBool = std::get<int>(condition.value) != 0;
  } else if (std::holds_alternative<float>(condition.value)) {
    condBool = std::get<float>(condition.value) != 0.0f;
  } else if (std::holds_alternative<double>(condition.value)) {
    condBool = std::get<double>(condition.value) != 0.0;
  } else {
    op.emitError("If condition must be bool, int, float, or double");
  }

  std::vector<TypedCppValue> results;

  if (condBool) {
    // Execute then region
    for (auto& bodyOp : op.getThenRegion().front().getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
        for (auto yieldOperand : yieldOp.getOperands()) {
          results.push_back(env.at(yieldOperand));
        }
      } else {
        visit(&bodyOp);
      }
    }
  } else if (!op.getElseRegion().empty()) {
    // Execute else region if it exists
    for (auto& bodyOp : op.getElseRegion().front().getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&bodyOp)) {
        for (auto yieldOperand : yieldOp.getOperands()) {
          results.push_back(env.at(yieldOperand));
        }
      } else {
        visit(&bodyOp);
      }
    }
  }

  // Store results
  for (auto [result, value] : llvm::zip(op.getResults(), results)) {
    env.insert_or_assign(result, value);
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
  for (auto initArg : op.getInits()) {
    iterArgs.push_back(env.at(initArg));
  }

  // Execute the loop
  for (int64_t i = lowerBound; i < upperBound; i += step) {
    // Set up induction variable and iter args in env
    env.erase(op.getInductionVar());
    env.insert({op.getInductionVar(), TypedCppValue(static_cast<int>(i))});

    for (auto [blockArg, iterArg] :
         llvm::zip(op.getRegionIterArgs(), iterArgs)) {
      env.erase(blockArg);
      env.insert({blockArg, iterArg});
    }

    // Walk the body and execute operations
    std::vector<TypedCppValue> yieldResults;
    for (auto& bodyOp : op.getBody()->getOperations()) {
      if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(&bodyOp)) {
        // Capture the yielded values
        for (auto yieldOperand : yieldOp.getOperands()) {
          yieldResults.push_back(env.at(yieldOperand));
        }
      } else {
        visit(&bodyOp);
      }
    }

    // Update iter args for next iteration
    iterArgs = std::move(yieldResults);
  }

  // Store final results
  for (auto [result, finalValue] : llvm::zip(op.getResults(), iterArgs)) {
    env.insert_or_assign(result, finalValue);
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

// OpenFHE binary operations
void Interpreter::visit(AddOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto lhsCt = std::get<CiphertextT>(lhs.value);
  auto rhsCt = std::get<CiphertextT>(rhs.value);
  TIME_OPERATION("Add", op.getOutput(), cc->EvalAdd(lhsCt, rhsCt));
}

void Interpreter::visit(AddPlainOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  // AddPlainOp can have ciphertext + plaintext in either order
  if (std::holds_alternative<CiphertextT>(lhs.value) &&
      std::holds_alternative<PlaintextT>(rhs.value)) {
    TIME_OPERATION("AddPlain", op.getOutput(),
                   cc->EvalAdd(std::get<CiphertextT>(lhs.value),
                               std::get<PlaintextT>(rhs.value)));
  } else if (std::holds_alternative<PlaintextT>(lhs.value) &&
             std::holds_alternative<CiphertextT>(rhs.value)) {
    TIME_OPERATION("AddPlain", op.getOutput(),
                   cc->EvalAdd(std::get<PlaintextT>(lhs.value),
                               std::get<CiphertextT>(rhs.value)));
  } else {
    op.emitError("AddPlainOp requires ciphertext and plaintext");
  }
}

void Interpreter::visit(SubOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto lhsCt = std::get<CiphertextT>(lhs.value);
  auto rhsCt = std::get<CiphertextT>(rhs.value);
  TIME_OPERATION("Sub", op.getOutput(), cc->EvalSub(lhsCt, rhsCt));
}

void Interpreter::visit(SubPlainOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  // SubPlainOp can have ciphertext - plaintext in either order
  if (std::holds_alternative<CiphertextT>(lhs.value) &&
      std::holds_alternative<PlaintextT>(rhs.value)) {
    TIME_OPERATION("SubPlain", op.getOutput(),
                   cc->EvalSub(std::get<CiphertextT>(lhs.value),
                               std::get<PlaintextT>(rhs.value)));
  } else if (std::holds_alternative<PlaintextT>(lhs.value) &&
             std::holds_alternative<CiphertextT>(rhs.value)) {
    TIME_OPERATION("SubPlain", op.getOutput(),
                   cc->EvalSub(std::get<PlaintextT>(lhs.value),
                               std::get<CiphertextT>(rhs.value)));
  } else {
    op.emitError("SubPlainOp requires ciphertext and plaintext");
  }
}

void Interpreter::visit(MulOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto lhsCt = std::get<CiphertextT>(lhs.value);
  auto rhsCt = std::get<CiphertextT>(rhs.value);
  TIME_OPERATION("Mul", op.getOutput(), cc->EvalMult(lhsCt, rhsCt));
}

void Interpreter::visit(MulNoRelinOp op) {
  auto lhs = env.at(op.getLhs());
  auto rhs = env.at(op.getRhs());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto lhsCt = std::get<CiphertextT>(lhs.value);
  auto rhsCt = std::get<CiphertextT>(rhs.value);
  TIME_OPERATION("MulNoRelin", op.getOutput(),
                 cc->EvalMultNoRelin(lhsCt, rhsCt));
}

void Interpreter::visit(MulPlainOp op) {
  auto ct = env.at(op.getCiphertext());
  auto pt = env.at(op.getPlaintext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  auto plaintext = std::get<PlaintextT>(pt.value);
  TIME_OPERATION("MulPlain", op.getOutput(),
                 cc->EvalMult(ciphertext, plaintext));
}

void Interpreter::visit(MulConstOp op) {
  auto ct = env.at(op.getCiphertext());
  auto constantVal = env.at(op.getConstant());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  int64_t constant = std::get<int>(constantVal.value);
  TIME_OPERATION("MulConst", op.getOutput(),
                 cc->EvalMult(ciphertext, constant));
}

// OpenFHE unary operations
void Interpreter::visit(NegateOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  TIME_OPERATION("Negate", op.getOutput(), cc->EvalNegate(ciphertext));
}

void Interpreter::visit(SquareOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  TIME_OPERATION("Square", op.getOutput(), cc->EvalSquare(ciphertext));
}

void Interpreter::visit(RelinOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  TIME_OPERATION("Relin", op.getOutput(), cc->Relinearize(ciphertext));
}

void Interpreter::visit(ModReduceOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  TIME_OPERATION("ModReduce", op.getOutput(), cc->ModReduce(ciphertext));
}

void Interpreter::visit(LevelReduceOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  auto levelToDrop = op.getLevelToDrop();
  TIME_OPERATION("LevelReduce", op.getOutput(),
                 cc->LevelReduce(ciphertext, nullptr, levelToDrop));
}

void Interpreter::visit(RotOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  auto index = op.getIndex().getValue().getSExtValue();
  TIME_OPERATION("Rot", op.getOutput(), cc->EvalRotate(ciphertext, index));
}

void Interpreter::visit(AutomorphOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto evalKey = env.at(op.getEvalKey());
  auto ciphertext = std::get<CiphertextT>(ct.value);
  // Note: AutomorphOp requires building a map with the eval key
  // For simplicity, we'll use index 0 as in the emitter
  std::map<uint32_t, EvalKeyT> evalKeyMap = {
      {0, std::get<EvalKeyT>(evalKey.value)}};
  TIME_OPERATION("Automorph", op.getOutput(),
                 cc->EvalAutomorphism(ciphertext, 0, evalKeyMap));
}

void Interpreter::visit(KeySwitchOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto evalKey = env.at(op.getEvalKey());
  auto ciphertext = std::get<CiphertextT>(ct.value);
  auto key = std::get<EvalKeyT>(evalKey.value);
  TIME_OPERATION("KeySwitch", op.getOutput(), cc->KeySwitch(ciphertext, key));
}

void Interpreter::visit(BootstrapOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  TIME_OPERATION("Bootstrap", op.getOutput(), cc->EvalBootstrap(ciphertext));
}

// OpenFHE encryption/decryption operations
void Interpreter::visit(EncryptOp op) {
  auto pt = env.at(op.getPlaintext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto plaintext = std::get<PlaintextT>(pt.value);
  // Note: EncryptOp takes an encryption key which could be public or private
  // For now, we'll need to handle both cases
  auto encKey = env.at(op.getEncryptionKey());
  if (std::holds_alternative<PublicKeyT>(encKey.value)) {
    auto publicKey = std::get<PublicKeyT>(encKey.value);
    TIME_OPERATION("Encrypt", op.getCiphertext(),
                   cc->Encrypt(publicKey, plaintext));
  } else if (std::holds_alternative<PrivateKeyT>(encKey.value)) {
    auto privateKey = std::get<PrivateKeyT>(encKey.value);
    TIME_OPERATION("Encrypt", op.getCiphertext(),
                   cc->Encrypt(privateKey, plaintext));
  } else {
    op.emitError("EncryptOp requires public or private key");
  }
}

void Interpreter::visit(DecryptOp op) {
  auto ct = env.at(op.getCiphertext());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto privateKey = env.at(op.getPrivateKey());
  auto ciphertext = std::get<CiphertextT>(ct.value);
  auto key = std::get<PrivateKeyT>(privateKey.value);
  PlaintextT plaintext;
  TIME_OPERATION_VOID("Decrypt", cc->Decrypt(key, ciphertext, &plaintext));
  env.insert_or_assign(op.getPlaintext(), plaintext);
}

void Interpreter::visit(MakePackedPlaintextOp op) {
  auto value = env.at(op.getValue());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  if (std::holds_alternative<std::vector<int>>(value.value)) {
    auto vec = std::get<std::vector<int>>(value.value);
    // Convert to int64_t vector as expected by OpenFHE
    std::vector<int64_t> vec64(vec.begin(), vec.end());
    TIME_OPERATION("MakePackedPlaintext", op.getPlaintext(),
                   cc->MakePackedPlaintext(vec64));
  } else {
    op.emitError("MakePackedPlaintextOp requires integer vector input");
  }
}

void Interpreter::visit(MakeCKKSPackedPlaintextOp op) {
  auto value = env.at(op.getValue());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  if (std::holds_alternative<std::vector<float>>(value.value)) {
    auto vec = std::get<std::vector<float>>(value.value);
    // Convert to double vector as expected by OpenFHE CKKS
    std::vector<double> vecDouble(vec.begin(), vec.end());
    TIME_OPERATION("MakeCKKSPackedPlaintext", op.getPlaintext(),
                   cc->MakeCKKSPackedPlaintext(vecDouble));
  } else if (std::holds_alternative<std::vector<double>>(value.value)) {
    auto vec = std::get<std::vector<double>>(value.value);
    // Already double, use directly
    TIME_OPERATION("MakeCKKSPackedPlaintext", op.getPlaintext(),
                   cc->MakeCKKSPackedPlaintext(vec));
  } else if (std::holds_alternative<std::vector<int>>(value.value)) {
    auto vec = std::get<std::vector<int>>(value.value);
    // Convert to double vector
    std::vector<double> vecDouble(vec.begin(), vec.end());
    TIME_OPERATION("MakeCKKSPackedPlaintext", op.getPlaintext(),
                   cc->MakeCKKSPackedPlaintext(vecDouble));
  } else {
    op.emitError("MakeCKKSPackedPlaintextOp requires numeric vector input");
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
  env.insert_or_assign(op.getResult(), cc);
}

void Interpreter::visit(GenRotKeyOp op) {
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto pk = std::get<PrivateKeyT>(env.at(op.getPrivateKey()).value);
  std::vector<int32_t> rotIndices(op.getIndices().begin(),
                                  op.getIndices().end());
  TIME_OPERATION_VOID("GenRotKey", cc->EvalRotateKeyGen(pk, rotIndices));
}

void Interpreter::visit(GenMulKeyOp op) {
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto pk = std::get<PrivateKeyT>(env.at(op.getPrivateKey()).value);
  TIME_OPERATION_VOID("GenMulKey", cc->EvalMultKeyGen(pk));
}

void Interpreter::visit(GenBootstrapKeyOp op) {
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto pk = std::get<PrivateKeyT>(env.at(op.getPrivateKey()).value);
  // Use full packing - ring dimension / 2
  auto numSlots = cc->GetRingDimension() / 2;
  TIME_OPERATION_VOID("GenBootstrapKey", cc->EvalBootstrapKeyGen(pk, numSlots));
}

void Interpreter::visit(SetupBootstrapOp op) {
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  std::vector<uint32_t> levelBudget = {
      static_cast<uint32_t>(
          op.getLevelBudgetEncode().getValue().getSExtValue()),
      static_cast<uint32_t>(
          op.getLevelBudgetDecode().getValue().getSExtValue())};
  TIME_OPERATION_VOID("SetupBootstrap", cc->EvalBootstrapSetup(levelBudget));
}

void Interpreter::visit(FastRotationOp op) {
  auto ct = env.at(op.getInput());
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto ciphertext = std::get<CiphertextT>(ct.value);
  auto index = op.getIndex().getZExtValue();
  auto digits =
      std::get<FastRotPrecompT>(env.at(op.getPrecomputedDigitDecomp()).value);
  auto m = 2 * cc->GetRingDimension();
  auto precomputeData = env.at(op.getPrecomputedDigitDecomp());
  TIME_OPERATION("FastRotation", op.getResult(),
                 cc->EvalFastRotation(ciphertext, index, m, digits));
}

void Interpreter::visit(FastRotationPrecomputeOp op) {
  auto cc = std::get<CryptoContextT>(env.at(op.getCryptoContext()).value);
  auto input = env.at(op.getInput());
  auto ciphertext = std::get<CiphertextT>(input.value);
  TIME_OPERATION("FastRotationPrecompute", op.getResult(),
                 cc->EvalFastRotationPrecompute(ciphertext));
}

void Interpreter::visit(lwe::RLWEDecodeOp op) {
  auto input = env.at(op.getInput());
  auto plaintext = std::get<PlaintextT>(input.value);

  bool isCKKS = llvm::isa<lwe::InverseCanonicalEncodingAttr>(op.getEncoding());

  // Check if the result is a tensor or a scalar
  if (auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType())) {
    // Get the size from the tensor type
    auto shape = tensorTy.getShape();
    // Find the non-unit dimension
    auto nonUnitDims = llvm::count_if(shape, [](auto dim) { return dim != 1; });
    if (nonUnitDims != 1) {
      op->emitError()
          << "Only 1D tensors with one non-unit dimension supported";
    }

    int64_t size = 0;
    for (auto dim : shape) {
      if (dim != 1) {
        size = dim;
        break;
      }
    }

    // Set the length of the plaintext
    plaintext->SetLength(size);

    // Get the packed values and convert to the appropriate type
    if (isCKKS) {
      // Get CKKS packed value (vector of complex<double>)
      auto ckksValues = plaintext->GetCKKSPackedValue();

      // Determine output precision based on result type
      auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
      bool outputIsDouble = false;
      if (resultType) {
        auto elemType = resultType.getElementType();
        if (auto floatType = dyn_cast<FloatType>(elemType)) {
          outputIsDouble = !floatType.isF32();
        }
      }

      if (outputIsDouble) {
        // Extract real parts as double
        std::vector<double> result;
        result.reserve(ckksValues.size());
        for (const auto& val : ckksValues) {
          result.push_back(val.real());
        }
        env.insert_or_assign(op.getResult(), std::move(result));
      } else {
        // Extract real parts and convert to float
        std::vector<float> result;
        result.reserve(ckksValues.size());
        for (const auto& val : ckksValues) {
          result.push_back(static_cast<float>(val.real()));
        }
        env.insert_or_assign(op.getResult(), std::move(result));
      }
    } else {
      // Get packed value (vector of int64_t)
      auto packedValues = plaintext->GetPackedValue();

      // Convert to int vector
      std::vector<int> result;
      result.reserve(packedValues.size());
      for (const auto& val : packedValues) {
        result.push_back(static_cast<int>(val));
      }

      env.insert_or_assign(op.getResult(), std::move(result));
    }
  } else {
    // Scalar result: get value at index 0
    if (isCKKS) {
      auto ckksValues = plaintext->GetCKKSPackedValue();
      // Determine output precision based on result type
      bool outputIsDouble = false;
      if (auto floatType = dyn_cast<FloatType>(op.getResult().getType())) {
        outputIsDouble = !floatType.isF32();
      }

      if (outputIsDouble) {
        double result = ckksValues[0].real();
        env.insert_or_assign(op.getResult(), result);
      } else {
        float result = static_cast<float>(ckksValues[0].real());
        env.insert_or_assign(op.getResult(), result);
      }
    } else {
      auto packedValues = plaintext->GetPackedValue();
      int result = static_cast<int>(packedValues[0]);
      env.insert_or_assign(op.getResult(), result);
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
}

OwningOpRef<ModuleOp> parse(MLIRContext* context, const std::string& mlirStr) {
  return parseSourceString<ModuleOp>(mlirStr, context);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
