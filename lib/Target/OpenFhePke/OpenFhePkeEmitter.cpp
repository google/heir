#include "lib/Target/OpenFhePke/OpenFhePkeEmitter.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <ios>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "include/cereal/archives/binary.hpp"           // from @cereal
#include "include/cereal/archives/json.hpp"             // from @cereal
#include "include/cereal/archives/portable_binary.hpp"  // from @cereal
#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"         // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/LogicalResult.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"               // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

namespace {

FailureOr<std::string> printFloatAttr(FloatAttr floatAttr) {
  if (!floatAttr.getType().isF32() && !floatAttr.getType().isF64()) {
    return failure();
  }

  SmallString<128> strValue;
  auto apValue = APFloat(floatAttr.getValueAsDouble());
  apValue.toString(strValue, /*FormatPrecision=*/0, /*FormatMaxPadding=*/15,
                   /*TruncateZero=*/true);
  return std::string(strValue);
}

FailureOr<std::string> getStringForConstant(Value value) {
  if (auto constantOp =
          dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
    auto valueAttr = constantOp.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
      return std::to_string(intAttr.getInt());
    } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
      return printFloatAttr(floatAttr);
    }
  }
  return failure();
}

// Returns true if the given array of sizes is contiguous.
bool isSingleContiguousSlice(ArrayRef<int64_t> sizes,
                             ArrayRef<int64_t> sourceShape) {
  // Find and drop leading and trailing unit sizes.
  ArrayRef<int64_t> range = sizes;
  const auto *nonUnitFrontIt =
      llvm::find_if(range, [](int64_t size) { return size != 1; });
  size_t numUnitFrontSizes = std::distance(range.begin(), nonUnitFrontIt);
  int64_t numDroppedFrontDims = std::min(numUnitFrontSizes + 1, range.size());
  const auto nonUnitBackIt = llvm::find_if(
      llvm::reverse(range), [](int64_t size) { return size != 1; });
  size_t numUnitBackSizes =
      std::distance(llvm::reverse(range).begin(), nonUnitBackIt);
  int64_t numDroppedBackDims = std::min(numUnitBackSizes, range.size());

  auto drop = [numDroppedFrontDims, numDroppedBackDims](auto range) {
    return range.drop_front(numDroppedFrontDims).drop_back(numDroppedBackDims);
  };

  // Check that the remaining size matches the source shape.
  return drop(range) == drop(sourceShape);
}

/// Print the integer element of a DenseElementsAttr.
static void printDenseIntElement(const APInt &value, raw_ostream &os,
                                 Type type) {
  if (type.isInteger(1))
    os << (value.getBoolValue() ? "true" : "false");
  else
    value.print(os, !type.isUnsignedInteger());
}

/// Print a floating point value in a way that the parser will be able to
/// round-trip losslessly.
static LogicalResult printFloatValue(const APFloat &apValue, raw_ostream &os) {
  assert(apValue.isFinite() && "expected finite value");
  SmallString<128> strValue;
  apValue.toString(strValue);
  // Parse back the stringized version and check that the value is equal
  // (i.e., there is no precision loss). If it is not, use the default format of
  // APFloat instead of the exponential notation.
  if (!APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue)) {
    return emitError(
        mlir::UnknownLoc(),
        llvm::formatv("Failed to print float value losslessly {0}", apValue));
    return failure();
  }

  os << strValue;
  return success();
}

FailureOr<std::string> printOneDimDenseElementsAttr(DenseElementsAttr attr) {
  auto type = attr.getType();
  auto elementType = attr.getElementType();
  if (!(elementType.isIntOrIndex() || llvm::isa<FloatType>(elementType))) {
    return failure();
  }

  std::string valueStr;
  llvm::raw_string_ostream ss(valueStr);
  auto printEltFn = [&](unsigned index) {
    if (elementType.isIntOrIndex()) {
      auto valueIt = attr.value_begin<APInt>();
      printDenseIntElement(*(valueIt + index), ss, elementType);
    } else {
      auto valueIt = attr.value_begin<APFloat>();
      if (failed(printFloatValue(*(valueIt + index), ss))) {
        return failure();
      }
    }
    return success();
  };
  if (attr.isSplat()) {
    if (failed(printEltFn(0))) {
      return failure();
    }
    return ss.str();
  }
  ss << "{";
  for (unsigned idx = 0, e = type.getNumElements(); idx != e; ++idx) {
    if (idx != 0) ss << ", ";
    if (failed(printEltFn(idx))) {
      return failure();
    }
  }
  ss << "}";
  return ss.str();
}

// Adds the given DenseElementsAttr to the weights map.
LogicalResult addWeightTo(DenseElementsAttr attr, std::string &name,
                          Weights *weights) {
  // Only double, float, and int_{8, 16, 32, 64}t are supported.
  return llvm::TypeSwitch<Type, LogicalResult>(attr.getElementType())
      .Case<Float32Type>([&](auto type) {
        std::vector<float> floats;
        for (auto value : attr.getValues<float>()) {
          floats.push_back(value);
        }
        weights->floats[name] = floats;
        return success();
      })
      .Case<Float64Type>([&](auto type) {
        std::vector<double> doubles;
        for (auto value : attr.getValues<double>()) {
          doubles.push_back(value);
        }
        weights->doubles[name] = doubles;
        return success();
      })
      .Case<IntegerType>([&](IntegerType type) {
        std::vector<int64_t> int64_ts;
        for (auto value : attr.getValues<APInt>()) {
          int64_ts.push_back(value.getSExtValue());
        }
        switch (type.getWidth()) {
          case 64:
            weights->int64_ts[name] = int64_ts;
            return success();
          case 32:
            weights->int32_ts[name] = {int64_ts.begin(), int64_ts.end()};
            return success();
          case 16:
            weights->int16_ts[name] = {int64_ts.begin(), int64_ts.end()};
            return success();
          case 8:
            weights->int8_ts[name] = {int64_ts.begin(), int64_ts.end()};
            return success();
          default:
            return failure();
        };
      })
      .Default([&](auto type) { return failure(); });
}

FailureOr<std::string> getWeightType(Type type) {
  auto result = llvm::TypeSwitch<Type, std::string>(type)
                    .Case<Float32Type>([&](auto type) { return "float"; })
                    .Case<Float64Type>([&](auto type) { return "double"; })
                    .Case<IntegerType>([&](auto type) {
                      return llvm::formatv("int{0}_t", type.getWidth());
                    })
                    .Default([&](auto type) { return ""; });
  if (result.empty()) {
    return failure();
  }
  return result;
}

}  // namespace

LogicalResult translateToOpenFhePke(Operation *op, llvm::raw_ostream &os,
                                    const OpenfheImportType &importType,
                                    const std::string &weightsFile,
                                    bool skipVectorResizing) {
  SelectVariableNames variableNames(op);
  OpenFhePkeEmitter emitter(os, &variableNames, importType, weightsFile,
                            skipVectorResizing);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult OpenFhePkeEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::CallOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Affine ops
          .Case<affine::AffineForOp, affine::AffineYieldOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::IndexCastOp, arith::ExtFOp, arith::RemSIOp,
                arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
                arith::CmpIOp, arith::SelectOp>(
              [&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ConcatOp, tensor::EmptyOp, tensor::InsertOp,
                tensor::InsertSliceOp, tensor::ExtractOp,
                tensor::ExtractSliceOp, tensor::SplatOp>(
              [&](auto op) { return printOperation(op); })
          // LWE ops
          .Case<lwe::RLWEDecodeOp, lwe::ReinterpretApplicationDataOp>(
              [&](auto op) { return printOperation(op); })
          // OpenFHE ops
          .Case<AddOp, AddPlainOp, SubOp, SubPlainOp, MulNoRelinOp, MulOp,
                MulPlainOp, SquareOp, NegateOp, MulConstOp, RelinOp,
                ModReduceOp, LevelReduceOp, RotOp, AutomorphOp, KeySwitchOp,
                EncryptOp, DecryptOp, GenParamsOp, GenContextOp, GenMulKeyOp,
                GenRotKeyOp, GenBootstrapKeyOp, MakePackedPlaintextOp,
                MakeCKKSPackedPlaintextOp, SetupBootstrapOp, BootstrapOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return emitError(op.getLoc(), "unable to find printer for op");
          });

  if (failed(status)) {
    return emitError(op.getLoc(),
                     llvm::formatv("Failed to translate op {0}", op.getName()));
  }
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(ModuleOp moduleOp) {
  OpenfheScheme scheme;
  if (moduleIsBGV(moduleOp)) {
    scheme = OpenfheScheme::BGV;
  } else if (moduleIsBFV(moduleOp)) {
    scheme = OpenfheScheme::BFV;
  } else if (moduleIsCKKS(moduleOp)) {
    scheme = OpenfheScheme::CKKS;
  } else {
    return emitError(moduleOp.getLoc(), "Missing scheme attribute on module");
  }

  os << getModulePrelude(scheme, importType_) << "\n";

  if (!weightsFile_.empty()) {
    os << getWeightsPrelude() << "\n";
  }
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  // Emit the weights file.
  if (!weightsFile_.empty()) {
    std::ofstream file(weightsFile_, std::ios::out | std::ios::binary);
    if (file.is_open()) {
      cereal::PortableBinaryOutputArchive archive(file);
      archive(weightsMap_);
      file.close();
    } else {
      return failure();
    }
  }
  return success();
}

bool OpenFhePkeEmitter::isDebugPort(StringRef debugPortName) {
  return debugPortName.rfind("__heir_debug") == 0;
}

StringRef OpenFhePkeEmitter::canonicalizeDebugPort(StringRef debugPortName) {
  if (isDebugPort(debugPortName)) {
    return "__heir_debug";
  }
  return debugPortName;
}

LogicalResult OpenFhePkeEmitter::printOperation(func::FuncOp funcOp) {
  if (funcOp.getNumResults() > 1) {
    return emitError(funcOp.getLoc(),
                     llvm::formatv("Only functions with a single return type "
                                   "are supported, but this function has ",
                                   funcOp.getNumResults()));
    return failure();
  }

  if (funcOp.getNumResults() == 1) {
    Type result = funcOp.getResultTypes()[0];
    if (failed(emitType(result, funcOp->getLoc()))) {
      return emitError(funcOp.getLoc(),
                       llvm::formatv("Failed to emit type {0}", result));
    }
  } else {
    os << "void";
  }

  os << " " << canonicalizeDebugPort(funcOp.getName()) << "(";
  os.indent();

  // Check the types without printing to enable failure outside of
  // commaSeparatedValues; maybe consider making commaSeparatedValues combine
  // the results into a FailureOr, like commaSeparatedTypes in tfhe_rust
  // emitter.
  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType(), arg.getLoc()))) {
      return emitError(funcOp.getLoc(),
                       llvm::formatv("Failed to emit type {0}", arg.getType()));
    }
  }

  if (funcOp.isDeclaration()) {
    // function declaration
    os << commaSeparatedTypes(funcOp.getArgumentTypes(), [&](Type type) {
      return convertType(type, funcOp->getLoc()).value();
    });
    // debug attribute map for debug call
    if (isDebugPort(funcOp.getName())) {
      os << ", const std::map<std::string, std::string>&";
    }
  } else {
    os << commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
      return convertType(value.getType(), funcOp->getLoc()).value() + " " +
             variableNames->getNameForValue(value);
    });
  }
  os.unindent();
  os << ")";

  // function declaration
  if (funcOp.isDeclaration()) {
    os << ";\n";
    return success();
  }

  os << " {\n";
  os.indent();

  if (!weightsFile_.empty() && !funcOp.getOps<arith::ConstantOp>().empty()) {
    os << llvm::formatv("Weights weights = GetWeightModule(\"{0}\");\n",
                        weightsFile_);
  }

  for (Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(func::CallOp op) {
  if (op.getNumResults() > 1) {
    return emitError(op.getLoc(), "Only one return value supported");
  }

  // build debug attribute map for debug call
  auto debugAttrMapName = getDebugAttrMapName();
  if (isDebugPort(op.getCallee())) {
    os << "std::map<std::string, std::string> " << debugAttrMapName << ";\n";
    for (auto attr : op->getAttrs()) {
      // callee is also an attribute internally, skip it
      if (attr.getName().getValue() == "callee") {
        continue;
      }
      os << debugAttrMapName << "[\"" << attr.getName().getValue()
         << "\"] = \"";
      // Use AsmPrinter to print Attribute
      if (mlir::isa<StringAttr>(attr.getValue())) {
        os << mlir::cast<StringAttr>(attr.getValue()).getValue() << "\";\n";
      } else {
        os << attr.getValue() << "\";\n";
      }
    }
    auto ciphertext = op->getOperand(op->getNumOperands() - 1);
    os << debugAttrMapName << R"(["asm.is_block_arg"] = ")"
       << isa<BlockArgument>(ciphertext) << "\";\n";
    if (auto *definingOp = ciphertext.getDefiningOp()) {
      os << debugAttrMapName << R"(["asm.op_name"] = ")"
         << definingOp->getName() << "\";\n";
    }
    // Use AsmPrinter to print Value
    os << debugAttrMapName << R"(["asm.result_ssa_format"] = ")" << ciphertext
       << "\";\n";
  }

  if (op.getNumResults() != 0) {
    emitAutoAssignPrefix(op.getResult(0));
  }

  os << canonicalizeDebugPort(op.getCallee()) << "(";
  os << commaSeparatedValues(op.getOperands(), [&](Value value) {
    return variableNames->getNameForValue(value);
  });
  // pass debug attribute map
  if (isDebugPort(op.getCallee())) {
    os << ", " << debugAttrMapName;
  }
  os << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(func::ReturnOp op) {
  if (op.getNumOperands() != 1) {
    return emitError(op.getLoc(), "Only one return value supported");
  }
  os << "return " << variableNames->getNameForValue(op.getOperands()[0])
     << ";\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(affine::AffineForOp op) {
  if (!op.hasConstantBounds()) {
    return emitError(op.getLoc(), "Only constant bounds are supported");
  }

  for (auto i = 0; i < op.getNumRegionIterArgs(); ++i) {
    // Assign the loop's results to use as initial iter args. We must use
    // non-const types so that the loop body can modify them.
    Value result = op.getResults()[i];
    Value operand = op.getOperands()[i];
    Value iterArg = op.getRegionIterArgs()[i];
    Value yieldedValue = op.getYieldedValues()[i];

    // Map the region's iter args to the result names.
    // Map the yielded value names to the result names.
    variableNames->mapValueNameToValue(iterArg, result);
    variableNames->mapValueNameToValue(yieldedValue, result);
    mutableValues.insert(op.getRegionIterArgs()[i]);
    mutableValues.insert(op.getYieldedValues()[i]);

    if (variableNames->contains(result) && variableNames->contains(operand) &&
        variableNames->getNameForValue(result) ==
            variableNames->getNameForValue(operand)) {
      // This occurs in cases where the loop is inserting into a tensor and
      // passing it along as an inter arg.
      continue;
    }

    if (failed(emitTypedAssignPrefix(result, op.getLoc(),
                                     /*constant=*/false))) {
      return emitError(
          op.getLoc(),
          llvm::formatv("Failed to emit typed assign prefix for {}", result));
    }
    os << variableNames->getNameForValue(operand);
    if (!isa<ShapedType>(result.getType())) {
      // Note that for vector types we don't need to clone.
      os << "->Clone()";
    }
    os << ";\n";
  }

  os << llvm::formatv("for (auto {0} = {1}; {0} < {2}; ++{0}) {{\n",
                      variableNames->getNameForValue(op.getInductionVar()),
                      op.getConstantLowerBound(), op.getConstantUpperBound());
  os.indent();
  for (Operation &op : *op.getBody()) {
    if (failed(translate(op))) {
      return op.emitOpError() << "Failed to translate for loop block";
    }
  }

  os.unindent();
  os << "}\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(affine::AffineYieldOp op) {
  // Assume all yielded loop values have already been assigned.
  return success();
}

void OpenFhePkeEmitter::emitAutoAssignPrefix(Value result) {
  // If the result values are iter args of a region, then avoid using a auto
  // assign prefix.
  if (!mutableValues.contains(result)) {
    //  Use const auto& because most OpenFHE API methods would
    // perform a copy if using a plain `auto`.
    os << "const auto& ";
  }
  os << variableNames->getNameForValue(result) << " = ";
}

LogicalResult OpenFhePkeEmitter::emitTypedAssignPrefix(Value result,
                                                       Location loc,
                                                       bool constant) {
  if (!mutableValues.contains(result)) {
    if (failed(emitType(result.getType(), loc, constant))) {
      return failure();
    }
    os << " ";
  }
  os << variableNames->getNameForValue(result) << " = ";
  return success();
}

LogicalResult OpenFhePkeEmitter::printEvalMethod(
    ::mlir::Value result, ::mlir::Value cryptoContext,
    ::mlir::ValueRange nonEvalOperands, std::string_view op) {
  emitAutoAssignPrefix(result);

  os << variableNames->getNameForValue(cryptoContext) << "->" << op << "(";
  os << commaSeparatedValues(nonEvalOperands, [&](Value value) {
    return variableNames->getNameForValue(value);
  });
  os << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(AddOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalAdd");
}

LogicalResult OpenFhePkeEmitter::printOperation(AddPlainOp op) {
  // OpenFHE defines an overload for EvalAdd to work on both plaintext and
  // ciphertext inputs.
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalAdd");
}

LogicalResult OpenFhePkeEmitter::printOperation(SubOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalSub");
}

LogicalResult OpenFhePkeEmitter::printOperation(SubPlainOp op) {
  // OpenFHE defines an overload for EvalSub to work on both plaintext and
  // ciphertext inputs.
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalSub");
}

LogicalResult OpenFhePkeEmitter::printOperation(MulNoRelinOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalMultNoRelin");
}

LogicalResult OpenFhePkeEmitter::printOperation(MulOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getLhs(), op.getRhs()}, "EvalMult");
}

LogicalResult OpenFhePkeEmitter::printOperation(MulPlainOp op) {
  // OpenFHE defines an overload for EvalMult to work on both plaintext and
  // ciphertext inputs.
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext(), op.getPlaintext()}, "EvalMult");
}

LogicalResult OpenFhePkeEmitter::printOperation(MulConstOp op) {
  // OpenFHE defines an overload for EvalMult to work on constant inputs,
  // but only for some schemes.
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext(), op.getConstant()}, "EvalMult");
}

LogicalResult OpenFhePkeEmitter::printOperation(NegateOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "EvalNegate");
}

LogicalResult OpenFhePkeEmitter::printOperation(SquareOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "EvalSquare");
}

LogicalResult OpenFhePkeEmitter::printOperation(RelinOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "Relinearize");
}

LogicalResult OpenFhePkeEmitter::printOperation(ModReduceOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "ModReduce");
}

LogicalResult OpenFhePkeEmitter::printOperation(LevelReduceOp op) {
  emitAutoAssignPrefix(op.getResult());

  os << variableNames->getNameForValue(op.getCryptoContext()) << "->"
     << "LevelReduce" << "(";
  os << commaSeparatedValues({op.getCiphertext()}, [&](Value value) {
    return variableNames->getNameForValue(value);
  });
  os << ", nullptr, ";
  os << op.getLevelToDrop() << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(RotOp op) {
  emitAutoAssignPrefix(op.getResult());

  os << variableNames->getNameForValue(op.getCryptoContext()) << "->"
     << "EvalRotate" << "("
     << variableNames->getNameForValue(op.getCiphertext()) << ", "
     << op.getIndex().getValue() << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(AutomorphOp op) {
  // EvalAutomorphism has a bit of a strange function signature in OpenFHE:
  //
  //     EvalAutomorphism(
  //       ConstCiphertext<DCRTPoly> ciphertext,
  //       int32_t i,
  //       const std::map<int32_t, EvalKey<DCRTPoly>>& evalKeyMap
  //     )
  //
  // Here i is an index to evalKeyMap, but no other data from evalKeyMap is
  // used. To match the API, we emit code that just creates a single-entry map
  // locally before calling EvalAutomorphism.
  //
  // This would probably be an easy upstream fix to add a specialized function
  // call if it becomes necessary.
  std::string mapName =
      variableNames->getNameForValue(op.getResult()) + "evalkeymap";
  auto result = convertType(op.getEvalKey().getType(), op->getLoc());
  os << "std::map<uint32_t, " << result << "> " << mapName << " = {{0, "
     << variableNames->getNameForValue(op.getEvalKey()) << "}};\n";

  emitAutoAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getCryptoContext())
     << "->EvalAutomorphism(";
  os << variableNames->getNameForValue(op.getCiphertext()) << ", 0, " << mapName
     << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(KeySwitchOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext(), op.getEvalKey()}, "KeySwitch");
}

LogicalResult OpenFhePkeEmitter::printOperation(BootstrapOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "EvalBootstrap");
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    // Constant integers may be unused if their uses directly output the
    // constant value (e.g. tensor.insert and tensor.extract use the defining
    // constant values of indices if available).
    os << "[[maybe_unused]] ";
    if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc()))) {
      return failure();
    }
    os << intAttr.getValue() << ";\n";
  } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    if (failed(emitTypedAssignPrefix(op.getResult(), op->getLoc()))) {
      return failure();
    }
    auto floatStr = printFloatAttr(floatAttr);
    if (failed(floatStr)) {
      return emitError(
          op.getLoc(),
          llvm::formatv("Failed to print floatAttr {0}", floatAttr));
    }
    os << floatStr.value() << ";\n";
  } else if (auto denseElementsAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    // TODO(#913): This is a simplifying assumption on the layout of the
    // multi-dimensional when there is only one non-unit dimension.
    // Prints all dense elements attribute as a flattened vector.
    ShapedType flattenedType =
        RankedTensorType::get({denseElementsAttr.getNumElements()},
                              denseElementsAttr.getType().getElementType());
    auto flattenedElementsAttr = denseElementsAttr.reshape(flattenedType);
    if (failed(emitType(flattenedElementsAttr.getType(), op.getLoc()))) {
      return failure();
    }

    auto name = variableNames->getNameForValue(op.getResult());
    os << " " << name;
    auto result = printOneDimDenseElementsAttr(flattenedElementsAttr);
    if (failed(result)) {
      return failure();
    }
    if (denseElementsAttr.isSplat()) {
      os << "(" << flattenedElementsAttr.getNumElements() << ", "
         << result.value() << ");\n";
    } else if (!weightsFile_.empty()) {
      if (failed(addWeightTo(flattenedElementsAttr, name, &weightsMap_))) {
        return emitError(
            op.getLoc(),
            llvm::formatv("Failed to add weight for type {0}", flattenedType));
      }
      auto weightType = getWeightType(flattenedType.getElementType());
      if (failed(weightType)) {
        return emitError(op.getLoc(),
                         llvm::formatv("Failed to get weight type for type {0}",
                                       flattenedType));
      }
      os << llvm::formatv(" = weights.{0}s[\"{1}\"];\n", weightType.value(),
                          name);
    } else {
      os << " = " << result.value() << ";\n";
    }
    return success();
  } else {
    return op.emitError() << "Unsupported constant type "
                          << valueAttr.getType();
  }
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::ExtSIOp op) {
  // OpenFHE has a convention that all inputs to MakePackedPlaintext are
  // std::vector<int64_t>, so earlier stages in the pipeline emit typecasts

  std::string inputVarName = variableNames->getNameForValue(op.getOperand());
  std::string resultVarName = variableNames->getNameForValue(op.getResult());

  // If it's a vector<int**_t>, we can use a copy constructor to upcast.
  if (auto tensorTy = dyn_cast<RankedTensorType>(op.getOperand().getType())) {
    os << "std::vector<int64_t> " << resultVarName << "(std::begin("
       << inputVarName << "), std::end(" << inputVarName << "));\n";
  } else {
    return op.emitOpError() << "Unsupported input type";
  }

  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::ExtUIOp op) {
  // OpenFHE has a convention that all inputs to MakePackedPlaintext are
  // std::vector<int64_t>, so earlier stages in the pipeline emit typecasts.
  // Unsigned extension should only be used for booleans.
  assert(
      getElementTypeOrSelf(op.getOperand().getType()).getIntOrFloatBitWidth() ==
          1 &&
      "expected boolean type for extui");

  std::string inputVarName = variableNames->getNameForValue(op.getOperand());
  std::string resultVarName = variableNames->getNameForValue(op.getResult());

  // If it's a vector<int**_t>, we can use a copy constructor to upcast.
  if (auto tensorTy = dyn_cast<RankedTensorType>(op.getOperand().getType())) {
    os << "std::vector<int64_t> " << resultVarName << "(std::begin("
       << inputVarName << "), std::end(" << inputVarName << "));\n";
  } else {
    return op.emitOpError() << "Unsupported input type";
  }

  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::ExtFOp op) {
  // OpenFHE has a convention that all inputs to MakeCKKSPackedPlaintext are
  // std::vector<double>, so earlier stages in the pipeline emit typecasts

  std::string inputVarName = variableNames->getNameForValue(op.getOperand());
  std::string resultVarName = variableNames->getNameForValue(op.getResult());

  // If it's a vector<float>, we can use a copy constructor to upcast.
  if (auto tensorTy = dyn_cast<RankedTensorType>(op.getOperand().getType())) {
    os << "std::vector<double> " << resultVarName << "(std::begin("
       << inputVarName << "), std::end(" << inputVarName << "));\n";
  } else {
    return op.emitOpError() << "Unsupported input type";
  }

  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::IndexCastOp op) {
  Type outputType = op.getOut().getType();
  if (failed(emitTypedAssignPrefix(op.getResult(), op->getLoc()))) {
    return failure();
  }
  os << "static_cast<";
  if (failed(emitType(outputType, op->getLoc()))) {
    return op.emitOpError() << "Unsupported index_cast op";
  }
  os << ">(" << variableNames->getNameForValue(op.getIn()) << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printBinaryOp(Operation *op, ::mlir::Value lhs,
                                               ::mlir::Value rhs,
                                               std::string_view opName) {
  assert(op->getNumResults() == 1 && "Expected single-result op!");
  if (failed(emitTypedAssignPrefix(op->getResult(0), op->getLoc(), true)))
    return failure();
  os << variableNames->getNameForValue(lhs) << " " << opName << " "
     << variableNames->getNameForValue(rhs) << ";\n";
  return success();
}

// Lowerings of ops like affine.apply involve scalar cleartext types
LogicalResult OpenFhePkeEmitter::printOperation(arith::AddIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "+");
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::MulIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "*");
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::SubIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "-");
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::DivSIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "/");
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::RemSIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "%");
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::SelectOp op) {
  if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc(), true)))
    return failure();

  os << variableNames->getNameForValue(op.getCondition()) << " ? "
     << variableNames->getNameForValue(op.getTrueValue()) << " : "
     << variableNames->getNameForValue(op.getFalseValue()) << ";\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(arith::CmpIOp op) {
  switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), "==");
    case arith::CmpIPredicate::ne:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), "!=");
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), "<");
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), "<=");
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), ">");
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), ">=");
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

LogicalResult OpenFhePkeEmitter::printOperation(tensor::ConcatOp op) {
  // concat dim(0) %foo, %foo, ...
  // lower to a loop
  auto operandType = cast<RankedTensorType>(op.getOperands()[0].getType());
  auto resultType = op.getResult().getType();
  std::string varName = variableNames->getNameForValue(op.getResult());
  if (resultType.getRank() != 1 || operandType.getRank() != 1) {
    return failure();
  }
  // std::vector<8192> result;
  if (failed(emitType(resultType, op->getLoc()))) {
    return failure();
  }
  os << " " << varName << ";\n";

  if (llvm::all_equal(op.getOperands())) {
    std::string operandName =
        variableNames->getNameForValue(op.getOperands()[0]);
    int64_t numRepeats =
        resultType.getNumElements() / operandType.getNumElements();
    // for (int i = 0; i < numRepeats; ++i) {
    os << "for (int i = 0; i < " << numRepeats << "; ++i) {\n";
    os.indent();

    // result.insert(result.end(), foo.begin(), foo.end());
    os << varName << ".insert(" << varName << ".end(), " << operandName
       << ".begin(), " << operandName << ".end());\n";

    os.unindent();
    os << "}\n";
    return success();
  }

  // More complicated concat ops are not supported yet. The earlier lowerings
  // should just produce concat for lack of a "repeat" op. Maybe we should make
  // a tensor_ext.repeat op?
  return failure();
}

LogicalResult OpenFhePkeEmitter::printOperation(tensor::EmptyOp op) {
  // std::vector<std::vector<CiphertextT>> result(dim0,
  // std::vector<CiphertextT>(dim1)); initStr = (dim1) initStr = (dim0,
  // std::vector<CiphertextT>{initStr})
  RankedTensorType resultTy = op.getResult().getType();
  auto elementTy = convertType(resultTy.getElementType(), op.getLoc());
  if (failed(elementTy)) {
    return failure();
  }
  if (failed(emitType(resultTy, op->getLoc()))) {
    return failure();
  }
  os << " " << variableNames->getNameForValue(op.getResult());
  std::string initStr = llvm::formatv("({0})", resultTy.getShape().back());
  for (auto dim :
       llvm::reverse(op.getResult().getType().getShape().drop_back(1))) {
    initStr = llvm::formatv("({0}, std::vector<{1}>{2})", dim,
                            elementTy.value(), initStr);
  }
  os << initStr << ";\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(tensor::ExtractOp op) {
  // const auto& v1 = in[0, 1];
  if (isa<lwe::NewLWECiphertextType, lwe::LWECiphertextType>(
          op.getResult().getType())) {
    emitAutoAssignPrefix(op.getResult());
  } else {
    if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc(), true)))
      return failure();
  }
  os << variableNames->getNameForValue(op.getTensor());
  os << "[";
  os << flattenIndexExpression(
      op.getTensor().getType(), op.getIndices(), [&](Value value) {
        auto constantStr = getStringForConstant(value);
        return constantStr.value_or(variableNames->getNameForValue(value));
      });
  os << "]";
  os << ";\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::extractRowFromMatrix(
    tensor::ExtractSliceOp op) {
  // Must be single contiguous slice with constant sizes and strides.
  if (llvm::any_of(op.getStaticSizes(),
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    return op.emitError() << "expected static sizes";
  }
  if (!llvm::all_of(op.getStaticStrides(),
                    [](int64_t size) { return size == 1; })) {
    return op.emitError() << "expected stride 1";
  }
  if (!isSingleContiguousSlice(op.getStaticSizes(),
                               op.getSourceType().getShape())) {
    return op.emitError() << "expected single contiguous slice";
  }
  // Only handle the case of extracting a single row from a 2-D tensor.
  // Offsets are expected to be [%val, 0]
  if (op.getMixedOffsets().size() != 2 || op.getStaticOffsets()[1] != 0) {
    return op.emitError() << "only support extracting one row from a 2D tensor";
  }

  std::string inputVarName = variableNames->getNameForValue(op.getSource());
  std::string resultVarName = variableNames->getNameForValue(op.getResult());
  auto elementType =
      convertType(op.getSourceType().getElementType(), op->getLoc());
  if (failed(elementType)) {
    return failure();
  }

  std::string flattenStart;
  if (op.isDynamicOffset(0)) {
    flattenStart = llvm::formatv(
        "{0} * {1}", variableNames->getNameForValue(op.getDynamicOffset(0)),
        op.getSourceType().getShape()[0]);
  } else {
    flattenStart = llvm::formatv("{0} * {1}", op.getStaticOffset(0),
                                 op.getSourceType().getShape()[0]);
  }
  auto flattenEnd = llvm::formatv("{0} + {1}", flattenStart,
                                  op.getResultType().getNumElements());
  os << "std::vector<" << elementType.value() << "> " << resultVarName
     << "(std::begin(" << inputVarName << ") + " << flattenStart
     << ", std::begin(" << inputVarName << ") + " << flattenEnd << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(tensor::ExtractSliceOp op) {
  RankedTensorType resultType = op.getResult().getType();

  // Otherwise we need a loop
  // For now only support 1D output tensors
  if (resultType.getRank() != 1) {
    return op.emitError() << "OpenFHE emitter only supports 1D output tensors "
                             "for extract_slice";
  }

  // If the input is a matrix, we can extract a row.
  if (resultType.getRank() != op.getSourceType().getRank()) {
    return extractRowFromMatrix(op);
  }

  std::string resultName = variableNames->getNameForValue(op.getResult());
  std::string sourceName = variableNames->getNameForValue(op.getSource());
  // std::vector<ty> result;
  if (failed(emitType(resultType, op->getLoc()))) {
    return failure();
  }
  os << " " << resultName << "(" << resultType.getNumElements() << ");\n";

  // If everything is 1D and has stride 1, we can use std::copy
  //
  // std::copy(source.begin() + offset, source.begin() + offset + size,
  // result.begin());
  if (resultType.getRank() == 1 && op.getSourceType().getRank() == 1 &&
      llvm::all_of(op.getStaticStrides(),
                   [](int64_t stride) { return stride == 1; })) {
    auto offset = op.getStaticOffsets()[0];
    auto size = op.getStaticSizes()[0];
    os << "std::copy(";
    os << sourceName << ".begin()" << " + " << offset << ", ";
    os << sourceName << ".begin()" << " + " << offset << " + " << size << ", ";
    os << resultName << ".begin());\n";
    return success();
  }

  return failure();
}

LogicalResult OpenFhePkeEmitter::printOperation(tensor::InsertSliceOp op) {
  RankedTensorType resultType = op.getResult().getType();
  if (resultType.getRank() != op.getSourceType().getRank()) {
    return op.emitError(
        "OpenFHE emitter for InsertSliceOp only supports "
        "result rank equal to source rank");
  }

  // We could relax this by using previously declared SSA values for dynamic
  // offsets, sizes, and strides. But we have no use for it yet and I'm facing
  // a deadline, baby!
  if (op.getStaticOffsets().empty() || op.getStaticSizes().empty() ||
      op.getStaticStrides().empty()) {
    return op.emitError() << "expected static offsets, sizes, and strides";
  }

  std::string destName = variableNames->getNameForValue(op.getDest());
  std::string sourceName = variableNames->getNameForValue(op.getSource());
  // The result tensor is materialized to the destination of the insert
  variableNames->mapValueNameToValue(op.getResult(), op.getDest());

  // If we have a 1D source and target tensor, and the strides are 1,
  // we can use std::copy
  //
  // std::copy(source.begin(), source.end(), dest.begin() + offset);
  if (resultType.getRank() == 1 && op.getSourceType().getRank() == 1 &&
      llvm::all_of(op.getStaticStrides(),
                   [](int64_t stride) { return stride == 1; })) {
    os << "std::copy(";
    os << sourceName << ".begin(), ";
    os << sourceName << ".end(), ";
    os << destName << ".begin() + " << op.getStaticOffsets()[0] << ");\n";
    return success();
  }

  SmallVector<int64_t> offsets = SmallVector<int64_t>(op.getStaticOffsets());
  SmallVector<int64_t> sizes = SmallVector<int64_t>(op.getStaticSizes());
  SmallVector<int64_t> strides = SmallVector<int64_t>(op.getStaticStrides());

  // Loop nest to copy the right values
  SmallVector<std::string> sourceIndexNames;
  SmallVector<std::string> destIndexNames;
  for (int nestLevel = 0; nestLevel < offsets.size(); nestLevel++) {
    std::string sourceIndexName = sourceName + "_" + std::to_string(nestLevel);
    std::string destIndexName = destName + "_" + std::to_string(nestLevel);
    sourceIndexNames.push_back(sourceIndexName);
    destIndexNames.push_back(destIndexName);
    // Initialize the source index to zero, since it is simple, note this must
    // happen outside the loop
    os << "int64_t " << sourceIndexName << " = 0;\n";
    os << "for (int64_t " << destIndexName << " = " << offsets[nestLevel]
       << "; " << destIndexName << " < "
       << offsets[nestLevel] + sizes[nestLevel] * strides[nestLevel] << "; "
       << destIndexName << " += " << strides[nestLevel] << ") {\n";
    os.indent();
  }

  // Now we're in the innermost loop nest, do the assignment
  os << destName << "[" << llvm::join(destIndexNames, "][")
     << "] = " << sourceName << "[" << llvm::join(sourceIndexNames, "][")
     << "];\n";

  // Now unindent and close the loop nest
  for (int nestLevel = offsets.size() - 1; nestLevel >= 0; nestLevel--) {
    // Also increment the source indices
    os << sourceIndexNames[nestLevel] << " += 1;\n";
    os.unindent();
    os << "}\n";
  }

  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(tensor::InsertOp op) {
  // For a tensor.insert MLIR statement, we assign the destination vector and
  // then map the result value to the destination value.
  // // %result = tensor.insert %scalar into %dest[%idx]
  // dest[idx] = scalar;
  os << variableNames->getNameForValue(op.getDest());
  os << "[";
  os << flattenIndexExpression(
      op.getResult().getType(), op.getIndices(), [&](Value value) {
        auto constantStr = getStringForConstant(value);
        return constantStr.value_or(variableNames->getNameForValue(value));
      });
  os << "]";
  os << " = " << variableNames->getNameForValue(op.getScalar()) << ";\n";

  variableNames->mapValueNameToValue(op.getResult(), op.getDest());
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(tensor::SplatOp op) {
  // std::vector<CiphertextType> result(num, value);
  auto result = op.getResult();
  if (failed(emitType(result.getType(), op->getLoc()))) {
    return failure();
  }
  if (result.getType().getRank() != 1) {
    return failure();
  }
  os << " " << variableNames->getNameForValue(result) << "("
     << result.getType().getNumElements() << ", "
     << variableNames->getNameForValue(op.getInput()) << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(
    lwe::ReinterpretApplicationDataOp op) {
  emitAutoAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getInput()) << ";\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(
    openfhe::MakePackedPlaintextOp op) {
  std::string inputVarName = variableNames->getNameForValue(op.getValue());
  FailureOr<Value> resultCC = getContextualCryptoContext(op.getOperation());
  if (failed(resultCC)) return resultCC;
  std::string cc = variableNames->getNameForValue(resultCC.value());

  if (skipVectorResizing_) {
    emitAutoAssignPrefix(op.getResult());
    os << cc << "->MakePackedPlaintext(" << inputVarName << ");\n";
    return success();
  }
  // cyclic repetition to mitigate openfhe zero-padding (#645)
  std::string filledPrefix =
      variableNames->getNameForValue(op.getResult()) + "_filled";
  std::string &inputVarFilledName = filledPrefix;
  std::string inputVarFilledLengthName = filledPrefix + "_n";

  os << "auto " << inputVarFilledLengthName << " = " << cc
     << "->GetCryptoParameters()->GetElementParams()->GetRingDimension() / "
        "2;\n";
  os << "auto " << inputVarFilledName << " = " << inputVarName << ";\n";
  os << inputVarFilledName << ".clear();\n";
  os << inputVarFilledName << ".reserve(" << inputVarFilledLengthName << ");\n";
  // inputVarFilledLengthName is unsigned
  os << "for (unsigned i = 0; i < " << inputVarFilledLengthName << "; ++i) {\n";
  os << "  " << inputVarFilledName << ".push_back(" << inputVarName << "[i % "
     << inputVarName << ".size()]);\n";
  os << "}\n";
  emitAutoAssignPrefix(op.getResult());
  os << cc << "->MakePackedPlaintext(" << inputVarFilledName << ");\n";

  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(
    openfhe::MakeCKKSPackedPlaintextOp op) {
  std::string inputVarName = variableNames->getNameForValue(op.getValue());
  FailureOr<Value> resultCC = getContextualCryptoContext(op.getOperation());
  if (failed(resultCC)) return resultCC;
  std::string cc = variableNames->getNameForValue(resultCC.value());

  if (skipVectorResizing_) {
    emitAutoAssignPrefix(op.getResult());
    os << variableNames->getNameForValue(resultCC.value())
       << "->MakeCKKSPackedPlaintext(" << inputVarName << ");\n";
    return success();
  }
  // cyclic repetition to mitigate openfhe zero-padding (#645)
  std::string filledPrefix =
      variableNames->getNameForValue(op.getResult()) + "_filled";
  std::string inputVarFilledName = filledPrefix;
  std::string inputVarFilledLengthName = filledPrefix + "_n";
  os << "auto " << inputVarFilledLengthName << " = " << cc
     << "->GetCryptoParameters()->GetElementParams()->GetRingDimension() / "
        "2;\n";
  os << "auto " << inputVarFilledName << " = " << inputVarName << ";\n";
  os << inputVarFilledName << ".clear();\n";
  os << inputVarFilledName << ".reserve(" << inputVarFilledLengthName << ");\n";
  os << "for (auto i = 0; i < " << inputVarFilledLengthName << "; ++i) {\n";
  os << "  " << inputVarFilledName << ".push_back(" << inputVarName << "[i % "
     << inputVarName << ".size()]);\n";
  os << "}\n";

  emitAutoAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(resultCC.value())
     << "->MakeCKKSPackedPlaintext(" << inputVarFilledName << ");\n";

  return success();
}

// Returns the unique non-unit dimension of a tensor and its rank.
// Returns failure if the tensor has more than one non-unit dimension.
// Utility function copied from SecretToCKKS.cpp
FailureOr<std::pair<unsigned, int64_t>> getNonUnitDimension(
    RankedTensorType tensorTy) {
  auto shape = tensorTy.getShape();

  if (llvm::count_if(shape, [](auto dim) { return dim != 1; }) != 1) {
    return failure();
  }

  unsigned nonUnitIndex = std::distance(
      shape.begin(), llvm::find_if(shape, [&](auto dim) { return dim != 1; }));

  return std::make_pair(nonUnitIndex, shape[nonUnitIndex]);
}

LogicalResult OpenFhePkeEmitter::printOperation(lwe::RLWEDecodeOp op) {
  // In OpenFHE a plaintext is already decoded by decrypt. The internal OpenFHE
  // implementation is simple enough (and dependent on currently-hard-coded
  // encoding choices) that we will eventually need to work at a lower level of
  // the API to support this operation properly.
  bool isCKKS = llvm::isa<lwe::InverseCanonicalEncodingAttr>(op.getEncoding());
  auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (tensorTy) {
    auto nonUnitDim = getNonUnitDimension(tensorTy);
    if (failed(nonUnitDim)) {
      return emitError(op.getLoc(), "Only 1D tensors supported");
    }
    // OpenFHE plaintexts must be manually resized to the decoded output size
    // via plaintext->SetLength(<size>);
    auto size = nonUnitDim.value().second;
    auto inputVarName = variableNames->getNameForValue(op.getInput());
    os << inputVarName << "->SetLength(" << size << ");\n";

    // Get the packed values in OpenFHE's type (vector of int_64t/complex/etc)
    std::string tmpVar =
        variableNames->getNameForValue(op.getResult()) + "_cast";
    os << "const auto& " << tmpVar << " = ";
    if (isCKKS) {
      os << inputVarName << "->GetCKKSPackedValue();\n";
    } else {
      os << inputVarName << "->GetPackedValue();\n";
    }

    // Convert to the intended type defined by the program
    auto outputVarName = variableNames->getNameForValue(op.getResult());
    if (failed(emitType(tensorTy, op->getLoc()))) {
      return failure();
    }
    if (isCKKS) {
      // need to drop the complex down to real:  first create the vector,
      os << " " << outputVarName << "(" << tmpVar << ".size());\n";
      // then use std::transform
      os << "std::transform(std::begin(" << tmpVar << "), std::end(" << tmpVar
         << "), std::begin(" << outputVarName
         << "), [](const std::complex<double>& c) { return c.real(); });\n";
    } else {
      // directly use a copy constructor
      os << " " << outputVarName << "(std::begin(" << tmpVar << "), std::end("
         << tmpVar << "));\n";
    }
    return success();
  }

  // By convention, a plaintext stores a scalar value in index 0
  auto result = emitTypedAssignPrefix(op.getResult(), op->getLoc());
  if (failed(result)) return result;
  os << variableNames->getNameForValue(op.getInput());
  if (isCKKS) {
    os << "->GetCKKSPackedValue()[0].real();\n";
  } else {
    os << "->GetPackedValue()[0];\n";
  }
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(EncryptOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getEncryptionKey(), op.getPlaintext()}, "Encrypt");
}

LogicalResult OpenFhePkeEmitter::printOperation(DecryptOp op) {
  // Decrypt asks for a pointer to an outparam for the output plaintext
  os << "PlaintextT " << variableNames->getNameForValue(op.getResult())
     << ";\n";

  os << variableNames->getNameForValue(op.getCryptoContext()) << "->Decrypt(";
  os << commaSeparatedValues(
      {op.getPrivateKey(), op.getCiphertext()},
      [&](Value value) { return variableNames->getNameForValue(value); });
  os << ", &" << variableNames->getNameForValue(op.getResult()) << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(GenParamsOp op) {
  auto paramsName = variableNames->getNameForValue(op.getResult());
  int64_t mulDepth = op.getMulDepthAttr().getValue().getSExtValue();
  int64_t plainMod = op.getPlainModAttr().getValue().getSExtValue();
  int64_t evalAddCount = op.getEvalAddCountAttr().getValue().getSExtValue();
  int64_t keySwitchCount = op.getKeySwitchCountAttr().getValue().getSExtValue();

  os << "CCParamsT " << paramsName << ";\n";
  // Essential parameters
  os << paramsName << ".SetMultiplicativeDepth(" << mulDepth << ");\n";
  if (plainMod != 0) {
    os << paramsName << ".SetPlaintextModulus(" << plainMod << ");\n";
  }
  // Optional parameters
  if (op.getRingDim() != 0) {
    os << paramsName << ".SetRingDim(" << op.getRingDim() << ");\n";
  }
  if (op.getBatchSize() != 0) {
    os << paramsName << ".SetBatchSize(" << op.getBatchSize() << ");\n";
  }
  // Modulus chain parameters
  if (op.getFirstModSize() != 0) {
    os << paramsName << ".SetFirstModSize(" << op.getFirstModSize() << ");\n";
  }
  if (op.getScalingModSize() != 0) {
    os << paramsName << ".SetScalingModSize(" << op.getScalingModSize()
       << ");\n";
  }
  // Advanced parameters
  if (evalAddCount != 0) {
    os << paramsName << ".SetEvalAddCount(" << evalAddCount << ");\n";
  }
  if (keySwitchCount != 0) {
    os << paramsName << ".SetKeySwitchCount(" << keySwitchCount << ");\n";
  }
  // Key switching technique parameters
  if (op.getDigitSize() != 0) {
    os << paramsName << ".SetDigitSize(" << op.getDigitSize() << ");\n";
  }
  if (op.getNumLargeDigits() != 0) {
    os << paramsName << ".SetNumLargeDigits(" << op.getNumLargeDigits()
       << ");\n";
  }
  // Relinearization technique parameters
  if (op.getMaxRelinSkDeg() != 0) {
    os << paramsName << ".SetMaxRelinSkDeg(" << op.getMaxRelinSkDeg() << ");\n";
  }
  // Option switches
  if (op.getInsecure()) {
    os << paramsName << ".SetSecurityLevel(lbcrypto::HEStd_NotSet);\n";
  }
  // For B/FV, OpenFHE supports EXTENDED encryption technique.
  if (op.getEncryptionTechniqueExtended()) {
    os << paramsName << ".SetEncryptionTechnique(EXTENDED);\n";
  }
  if (!op.getKeySwitchingTechniqueBV()) {
    // B/FV defaults to BV, to match HEIR parameter generation we need to
    // set it to HYBRID. Other schemes defaults to HYBRID.
    os << paramsName << ".SetKeySwitchTechnique(HYBRID);\n";
  } else {
    os << paramsName << ".SetKeySwitchTechnique(BV);\n";
  }
  if (op.getScalingTechniqueFixedManual()) {
    os << paramsName << ".SetScalingTechnique(FIXEDMANUAL);\n";
  }
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(GenContextOp op) {
  auto paramsName = variableNames->getNameForValue(op.getParams());
  auto contextName = variableNames->getNameForValue(op.getResult());

  os << "CryptoContextT " << contextName << " = GenCryptoContext(" << paramsName
     << ");\n";
  os << contextName << "->Enable(PKE);\n";
  os << contextName << "->Enable(KEYSWITCH);\n";
  os << contextName << "->Enable(LEVELEDSHE);\n";
  if (op.getSupportFHE()) {
    os << contextName << "->Enable(ADVANCEDSHE);\n";
    os << contextName << "->Enable(FHE);\n";
  }
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(GenMulKeyOp op) {
  auto contextName = variableNames->getNameForValue(op.getCryptoContext());
  auto privateKeyName = variableNames->getNameForValue(op.getPrivateKey());
  os << contextName << "->EvalMultKeyGen(" << privateKeyName << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(GenRotKeyOp op) {
  auto contextName = variableNames->getNameForValue(op.getCryptoContext());
  auto privateKeyName = variableNames->getNameForValue(op.getPrivateKey());

  std::vector<std::string> rotIndices;
  llvm::transform(op.getIndices(), std::back_inserter(rotIndices),
                  [](int64_t value) { return std::to_string(value); });

  os << contextName << "->EvalRotateKeyGen(" << privateKeyName << ", {";
  os << llvm::join(rotIndices, ", ");
  os << "});\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(GenBootstrapKeyOp op) {
  auto contextName = variableNames->getNameForValue(op.getCryptoContext());
  auto privateKeyName = variableNames->getNameForValue(op.getPrivateKey());
  // compiler can not determine slot num for now
  // full packing for CKKS, as we currently always full packing
  os << "auto numSlots = " << contextName << "->GetRingDimension() / 2;\n";
  os << contextName << "->EvalBootstrapKeyGen(" << privateKeyName
     << ", numSlots);\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(SetupBootstrapOp op) {
  auto contextName = variableNames->getNameForValue(op.getCryptoContext());
  os << contextName << "->EvalBootstrapSetup({";
  os << op.getLevelBudgetEncode().getValue() << ", ";
  os << op.getLevelBudgetDecode().getValue();
  os << "});\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::emitType(Type type, Location loc,
                                          bool constant) {
  auto result = convertType(type, loc, constant);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

OpenFhePkeEmitter::OpenFhePkeEmitter(raw_ostream &os,
                                     SelectVariableNames *variableNames,
                                     const OpenfheImportType &importType,
                                     const std::string &weightsFile,
                                     bool skipVectorResizing)
    : importType_(importType),
      os(os),
      variableNames(variableNames),
      weightsFile_(weightsFile),
      skipVectorResizing_(skipVectorResizing) {}
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
