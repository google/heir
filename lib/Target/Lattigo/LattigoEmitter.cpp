#include "lib/Target/Lattigo/LattigoEmitter.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Target/Lattigo/LattigoTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/Support/CommandLine.h"     // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"   // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/ManagedStatic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/IndentedOstream.h"   // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

LogicalResult translateToLattigo(Operation* op, llvm::raw_ostream& os,
                                 const std::string& packageName) {
  SelectVariableNames variableNames(op);
  std::string bufferedStr;
  llvm::raw_string_ostream strOs(bufferedStr);
  raw_indented_ostream bufferedOs(strOs);
  LattigoEmitter emitter(bufferedOs, &variableNames, packageName);
  LogicalResult result = emitter.translate(*op);

  // Now write the materialized prelude and body to the outstream.
  emitter.emitPrelude(os);
  os << strOs.str();
  return result;
}

LogicalResult LattigoEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp, func::CallOp>(
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
          .Case<tensor::ConcatOp, tensor::EmptyOp, tensor::ExtractOp,
                tensor::ExtractSliceOp, tensor::InsertOp, tensor::InsertSliceOp,
                tensor::FromElementsOp, tensor::SplatOp>(
              [&](auto op) { return printOperation(op); })
          // Lattigo ops
          .Case<
              // RLWE
              RLWENewEncryptorOp, RLWENewDecryptorOp, RLWENewKeyGeneratorOp,
              RLWEGenKeyPairOp, RLWEGenRelinearizationKeyOp, RLWEGenGaloisKeyOp,
              RLWENewEvaluationKeySetOp, RLWEEncryptOp, RLWEDecryptOp,
              RLWEDropLevelNewOp, RLWEDropLevelOp, RLWENegateNewOp,
              RLWENegateOp,
              // BGV
              BGVNewParametersFromLiteralOp, BGVNewEncoderOp, BGVNewEvaluatorOp,
              BGVNewPlaintextOp, BGVEncodeOp, BGVDecodeOp, BGVAddNewOp,
              BGVSubNewOp, BGVMulNewOp, BGVAddOp, BGVSubOp, BGVMulOp,
              BGVRelinearizeOp, BGVRescaleOp, BGVRotateColumnsOp,
              BGVRotateRowsOp, BGVRelinearizeNewOp, BGVRescaleNewOp,
              BGVRotateColumnsNewOp, BGVRotateRowsNewOp,
              // CKKS
              CKKSNewParametersFromLiteralOp, CKKSNewEncoderOp,
              CKKSNewEvaluatorOp, CKKSNewPlaintextOp, CKKSEncodeOp,
              CKKSDecodeOp, CKKSAddNewOp, CKKSSubNewOp, CKKSMulNewOp, CKKSAddOp,
              CKKSSubOp, CKKSMulOp, CKKSRelinearizeOp, CKKSRescaleOp,
              CKKSRotateOp, CKKSRelinearizeNewOp, CKKSRescaleNewOp,
              CKKSRotateNewOp>([&](auto op) { return printOperation(op); })
          .Default([&](Operation&) {
            return emitError(op.getLoc(), "unable to find printer for op");
          });

  if (failed(status)) {
    return emitError(op.getLoc(),
                     llvm::formatv("Failed to translate op {0}", op.getName()));
  }
  return success();
}

LogicalResult LattigoEmitter::printOperation(ModuleOp moduleOp) {
  prelude = "package " + packageName + "\n";

  // Defer prelude and imports until the end, since some ops may need extra
  // imports injected to the `imports` list dynamically as they are emitted.
  imports.insert(std::string(kRlweImport));
  if (moduleIsBGVOrBFV(moduleOp)) {
    imports.insert(std::string(kBgvImport));
  } else if (moduleIsCKKS(moduleOp)) {
    imports.insert(std::string(kMathImport));
    imports.insert(std::string(kCkksImport));
  } else {
    return moduleOp.emitError("Unknown scheme");
  }

  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult LattigoEmitter::printOperation(func::FuncOp funcOp) {
  // skip debug functions, should be defined in another file
  if (isDebugPort(funcOp.getName())) {
    return success();
  }

  // name and arg
  os << "func " << funcOp.getName() << "(";
  os << getCommaSeparatedNamesWithTypes(funcOp.getArguments());
  os << ") ";

  // return types
  auto resultTypesString = getCommaSeparatedTypes(funcOp.getResultTypes());
  if (failed(resultTypesString)) {
    return failure();
  }
  if (!resultTypesString->empty()) {
    os << "(";
    os << resultTypesString.value();
    os << ") ";
  }
  os << "{\n";
  os.indent();

  // body
  for (Block& block : funcOp.getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  os.unindent();
  os << "}\n";

  return success();
}

LogicalResult LattigoEmitter::printOperation(func::ReturnOp op) {
  os << "return ";
  os << getCommaSeparatedNames(op.getOperands());
  os << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(func::CallOp op) {
  // build debug attribute map for debug call
  auto debugAttrMapName = getDebugAttrMapName();
  if (isDebugPort(op.getCallee())) {
    os << debugAttrMapName << " := make(map[string]string)\n";
    for (auto attr : op->getAttrs()) {
      // callee is also an attribute internally, skip it
      if (attr.getName().getValue() == "callee") {
        continue;
      }
      os << debugAttrMapName << "[\"" << attr.getName().getValue()
         << "\"] = \"";
      // Use AsmPrinter to print Attribute
      if (mlir::isa<StringAttr>(attr.getValue())) {
        os << mlir::cast<StringAttr>(attr.getValue()).getValue() << "\"\n";
      } else {
        os << attr.getValue() << "\"\n";
      }
    }
    auto ciphertext = op->getOperand(op->getNumOperands() - 1);
    os << debugAttrMapName << R"(["asm.is_block_arg"] = ")"
       << isa<BlockArgument>(ciphertext) << "\"\n";
    if (auto* definingOp = ciphertext.getDefiningOp()) {
      os << debugAttrMapName << R"(["asm.op_name"] = ")"
         << definingOp->getName() << "\"\n";
    }
    // Use AsmPrinter to print Value
    os << debugAttrMapName << R"(["asm.result_ssa_format"] = ")" << ciphertext
       << "\"\n";
  }

  if (op.getNumResults() > 0) {
    os << getCommaSeparatedNames(op.getResults());
    os << " := ";
  }
  os << canonicalizeDebugPort(op.getCallee()) << "(";
  os << getCommaSeparatedNames(op.getOperands());
  // pass debug attribute map
  if (isDebugPort(op.getCallee())) {
    os << ", " << debugAttrMapName;
  }
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(affine::AffineForOp op) {
  if (!op.hasConstantBounds()) {
    return emitError(op.getLoc(), "Only constant bounds are supported");
  }

  for (auto i = 0; i < op.getNumRegionIterArgs(); ++i) {
    // Assign the loop's results to use as initial iter args.
    Value result = op.getResults()[i];
    Value operand = op.getOperands()[i];
    Value iterArg = op.getRegionIterArgs()[i];

    if (variableNames->contains(result) && variableNames->contains(operand) &&
        variableNames->getNameForValue(result) ==
            variableNames->getNameForValue(operand)) {
      // This occurs in cases where the loop is inserting into a tensor and
      // passing it along as an inter arg.
      continue;
    }

    os << getName(iterArg) << " := " << getName(operand) << "\n";
  }

  os << llvm::formatv("for {0} := int64({1}); {0} < {2}; {0} += {3} {{\n",
                      // Don't use getName here because it is guaranteed to be
                      // used by the loop update calls
                      variableNames->getNameForValue(op.getInductionVar()),
                      op.getConstantLowerBound(), op.getConstantUpperBound(),
                      op.getStepAsInt());
  os.indent();
  for (Operation& op : *op.getBody()) {
    if (failed(translate(op))) {
      return op.emitOpError() << "Failed to translate for loop block";
    }
  }

  // Now assign the final yielded results to the iter arg variable
  for (auto i = 0; i < op.getNumRegionIterArgs(); ++i) {
    // Assign the loop's results to use as initial iter args.
    Value iterArg = op.getRegionIterArgs()[i];
    Value yieldedResult = op.getYieldedValues()[i];

    os << getName(iterArg) << " = " << getName(yieldedResult) << "\n";
  }

  os.unindent();
  os << "}\n";

  // Now assign the final iterArgs to the loop results
  for (auto i = 0; i < op.getNumRegionIterArgs(); ++i) {
    Value result = op.getResults()[i];
    Value iterArg = op.getRegionIterArgs()[i];
    os << getName(result) << " := " << getName(iterArg) << "\n";
  }
  return success();
}

LogicalResult LattigoEmitter::printOperation(affine::AffineYieldOp op) {
  // Assume all yielded loop values have already been assigned.
  return success();
}

namespace {

template <typename T>
FailureOr<std::string> getStringForConstant(T value) {
  if constexpr (std::is_same_v<T, APInt>) {
    if (value.getBitWidth() == 1) {
      return std::string(value.getBoolValue() ? "true" : "false");
    }
    return std::to_string(value.getSExtValue());
  } else if constexpr (std::is_same_v<T, APFloat>) {
    // care about precision...see OpenfhePkeEmitter when we encounter problem
    std::string literalStr;
    llvm::raw_string_ostream literalOs(literalStr);
    value.print(literalOs);
    return literalOs.str();
  }
  return failure();
}

FailureOr<std::string> getStringForDenseElementAttr(
    DenseElementsAttr denseAttr) {
  // Splat is also handled in getValues
  SmallVector<std::string> values;
  if (succeeded(denseAttr.tryGetValues<APInt>())) {
    for (auto value : denseAttr.getValues<APInt>()) {
      auto constantString = getStringForConstant(value);
      if (failed(constantString)) {
        return failure();
      }
      values.push_back(*constantString);
    }
  } else if (succeeded(denseAttr.tryGetValues<APFloat>())) {
    for (auto value : denseAttr.getValues<APFloat>()) {
      auto constantString = getStringForConstant(value);
      if (failed(constantString)) {
        return failure();
      }
      values.push_back(*constantString);
    }
  } else {
    return failure();
  }
  return llvm::join(values, ", ");
}

}  // namespace

LogicalResult LattigoEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  auto type = valueAttr.getType();
  // GO use () for scalar: int64()
  std::string left = "(";
  std::string right = ")";
  if (isa<RankedTensorType>(type)) {
    // GO use {} for slice: []int64{}
    left = "{";
    right = "}";
  }
  auto typeString = convertType(valueAttr.getType());
  if (failed(typeString)) {
    return failure();
  }
  std::string valueString;
  valueString = *typeString + left;
  auto res =
      llvm::TypeSwitch<Attribute, LogicalResult>(valueAttr)
          .Case<IntegerAttr>([&](IntegerAttr intAttr) {
            auto constantString = getStringForConstant(intAttr.getValue());
            if (failed(constantString)) {
              return failure();
            }
            valueString += *constantString;
            return success();
          })
          .Case<FloatAttr>([&](FloatAttr floatAttr) {
            auto constantString = getStringForConstant(floatAttr.getValue());
            if (failed(constantString)) {
              return failure();
            }
            valueString += *constantString;
            return success();
          })
          .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
            // fill in the values
            auto constString = getStringForDenseElementAttr(denseAttr);
            if (failed(constString)) {
              return failure();
            }
            valueString += *constString;
            return success();
          })
          .Default([&](auto) { return failure(); });
  if (failed(res)) {
    return res;
  }
  valueString += right;
  os << getName(op.getResult()) << " := " << valueString << "\n";
  return success();
}

void LattigoEmitter::emitIf(const std::string& cond,
                            const std::function<void()>& trueBranch,
                            const std::function<void()>& falseBranch) {
  os << "if " << cond << " {\n";
  os.indent();
  trueBranch();
  os.unindent();
  os << "} else {\n";
  os.indent();
  falseBranch();
  os.unindent();
  os << "}\n";
}

LogicalResult LattigoEmitter::typecast(Value operand, Value result) {
  std::string inputVarName = getName(operand);

  // If it's a slice, upcast by creating a new slice and looping
  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    if (tensorTy.getRank() > 1) {
      return emitError(operand.getDefiningOp()->getLoc(),
                       "Unsupported cast; expected 1D tensor");
    }
    auto res = convertType(tensorTy.getElementType());
    if (failed(res)) {
      return failure();
    }
    std::string elementType = res.value();
    // Don't use getName because the usage of the variable is guaranteed
    std::string resultVarName = variableNames->getNameForValue(result);
    os << resultVarName << " := make([]" << elementType << ", ";
    os << tensorTy.getNumElements() << ")\n";

    // Now loop and apply the cast
    os << "for i, val := range " << inputVarName << " {\n";
    os.indent();
    os << resultVarName << "[i] = " << elementType << "(val)\n";
    os.unindent();
    os << "}\n";
  } else {
    auto res = convertType(result.getType());
    if (failed(res)) {
      return failure();
    }
    std::string resultTy = res.value();
    os << getName(result) << " := " << resultTy << "(" << inputVarName << ")\n";
  }

  return success();
}

LogicalResult LattigoEmitter::printOperation(arith::ExtSIOp op) {
  return typecast(op.getOperand(), op.getResult());
}

LogicalResult LattigoEmitter::printOperation(arith::ExtUIOp op) {
  // Unsigned extension should only be used for booleans.
  assert(
      getElementTypeOrSelf(op.getOperand().getType()).getIntOrFloatBitWidth() ==
          1 &&
      "expected boolean type for extui");

  // golang cannot directly convert booleans to integers, so we need to branch
  std::string inputVarName = getName(op.getOperand());

  // If it's a slice, upcast by creating a new slice and looping
  if (auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType())) {
    if (tensorTy.getRank() > 1) {
      return op.emitOpError()
             << "Unsupported input type for extui, expected 1D tensor";
    }
    auto res = convertType(tensorTy.getElementType());
    if (failed(res)) {
      return failure();
    }
    std::string elementType = res.value();
    // Don't use getName because the usage of the variable is guaranteed
    std::string resultVarName = variableNames->getNameForValue(op.getResult());
    os << resultVarName << " := make([]" << elementType << ", ";
    os << tensorTy.getNumElements() << ")\n";

    // Now loop and apply the cast
    // for i, val := range input {
    //   if val {
    //     result[i] = extendedType(1)
    //   } else {
    //     result[i] = extendedType(0)
    //   }
    // }
    os << "for i, val := range " << inputVarName << " {\n";
    os.indent();

    emitIf(
        "val",
        [&]() { os << resultVarName << "[i] = " << elementType << "(1)\n"; },
        [&]() { os << resultVarName << "[i] = " << elementType << "(0)\n"; });

    os.unindent();
    os << "}\n";
  } else {
    // A plain if to upcast a bool
    // var result extendedType
    // if operand {
    //   result = extendedType(1)
    // } else {
    //   result = extendedType(0)
    // }
    auto res = convertType(op.getResult().getType());
    if (failed(res)) {
      return failure();
    }
    std::string resultName = getName(op.getResult());
    std::string resultTy = res.value();

    os << "var " << resultName << " " << resultTy << "\n";
    emitIf(
        inputVarName,
        [&]() { os << resultName << " = " << resultTy << "(1)\n"; },
        [&]() { os << resultName << " = " << resultTy << "(0)\n"; });
  }

  return success();
}

LogicalResult LattigoEmitter::printOperation(arith::ExtFOp op) {
  return typecast(op.getOperand(), op.getResult());
}

LogicalResult LattigoEmitter::printOperation(arith::IndexCastOp op) {
  return typecast(op.getOperand(), op.getOut());
}

LogicalResult LattigoEmitter::printBinaryOp(Operation* op, ::mlir::Value lhs,
                                            ::mlir::Value rhs,
                                            std::string_view opName) {
  assert(op->getNumResults() == 1 && "Expected single-result op!");
  os << getName(op->getResult(0)) << " := " << getName(lhs) << " " << opName
     << " " << getName(rhs) << ";\n";
  return success();
}

// Lowerings of ops like affine.apply involve scalar cleartext types
LogicalResult LattigoEmitter::printOperation(arith::AddIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "+");
}

LogicalResult LattigoEmitter::printOperation(arith::MulIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "*");
}

LogicalResult LattigoEmitter::printOperation(arith::SubIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "-");
}

LogicalResult LattigoEmitter::printOperation(arith::DivSIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "/");
}

LogicalResult LattigoEmitter::printOperation(arith::RemSIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "%");
}

LogicalResult LattigoEmitter::printOperation(arith::SelectOp op) {
  std::string resultName = getName(op.getResult());
  auto res = convertType(op.getResult().getType());
  if (failed(res)) return failure();

  // Declare variable without assignment first, since go does not have a
  // ternary if.
  os << "var " << resultName << " " << res.value() << "\n";
  emitIf(
      getName(op.getCondition()),
      [&]() {
        os << resultName << " = " << getName(op.getTrueValue()) << "\n";
      },
      [&]() {
        os << resultName << " = " << getName(op.getFalseValue()) << "\n";
      });

  return success();
}

LogicalResult LattigoEmitter::printOperation(arith::CmpIOp op) {
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
  return failure();
}

LogicalResult LattigoEmitter::printOperation(tensor::ConcatOp op) {
  // Use slices.Concat which has the same semantics as 1D tensor.concat
  if (op.getResultType().getRank() != 1) {
    return op.emitError("Lattigo emitter for ConcatOp only supports rank 1");
  }
  imports.insert(std::string(kSlicesImport));
  SmallVector<std::string> operandNames = llvm::to_vector<4>(llvm::map_range(
      op.getInputs(), [&](Value value) { return getName(value); }));
  os << getName(op.getResult()) << " := slices.Concat("
     << llvm::join(operandNames, ", ") << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::EmptyOp op) {
  // Only support 1D tensors for now, initialize as a slice
  ShapedType resultType = op.getResult().getType();
  if (resultType.getRank() != 1) {
    return op.emitError("Lattigo emitter for ConcatOp only supports rank 1");
  }
  os << getName(op.getResult()) << " := make([]"
     << convertType(resultType.getElementType()) << ", "
     << resultType.getNumElements() << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::ExtractOp op) {
  os << getName(op.getResult()) << " := " << getName(op.getTensor()) << "[";
  os << flattenIndexExpression(
      op.getTensor().getType(), op.getIndices(),
      [&](Value value) { return variableNames->getNameForValue(value); });
  os << "]\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::ExtractSliceOp op) {
  // Not sure if golang has a strided slice operation like python,
  // so do it as a loop
  RankedTensorType resultType = op.getResult().getType();

  if (resultType.getRank() != op.getSourceType().getRank()) {
    return op.emitError(
        "Lattigo emitter for ExtractSliceOp only supports "
        "result rank equal to source rank");
  }

  // make an array to store the output
  //
  //   v := [5][6][7]int32{}
  //
  SmallVector<std::string> arrays;
  for (auto dim : resultType.getShape()) {
    arrays.push_back("[" + std::to_string(dim) + "]");
  }
  std::string resultName = getName(op.getResult(), /*force=*/true);
  std::string tmpName = resultName + "_array";
  os << tmpName << " := " << llvm::join(arrays, "")
     << convertType(resultType.getElementType()) << "{}\n";

  if (op.getStaticOffsets().empty() || op.getStaticSizes().empty() ||
      op.getStaticStrides().empty()) {
    return op.emitError() << "expected static offsets, sizes, and strides";
  }

  SmallVector<int64_t> offsets = SmallVector<int64_t>(op.getStaticOffsets());
  SmallVector<int64_t> sizes = SmallVector<int64_t>(op.getStaticSizes());
  SmallVector<int64_t> strides = SmallVector<int64_t>(op.getStaticStrides());

  // Loop nest to copy the right values
  SmallVector<std::string> sourceIndexNames;
  SmallVector<std::string> destIndexNames;
  for (int nestLevel = 0; nestLevel < offsets.size(); nestLevel++) {
    std::string sourceIndexName =
        resultName + "_source_" + std::to_string(nestLevel);
    std::string destIndexName =
        resultName + "_dest_" + std::to_string(nestLevel);
    sourceIndexNames.push_back(sourceIndexName);
    destIndexNames.push_back(destIndexName);
    // Initialise the destination index to zero, since it is simple, note this
    // must happen outside the loop.
    os << destIndexName << " := 0\n";
    os << "for " << sourceIndexName << " := " << offsets[nestLevel] << "; "
       << sourceIndexName << " < "
       << offsets[nestLevel] + sizes[nestLevel] * strides[nestLevel] << "; "
       << sourceIndexName << " += " << strides[nestLevel] << " {\n";
    os.indent();
  }

  // Now we're in the innermost loop nest, do the assignment
  os << tmpName << "[" << llvm::join(destIndexNames, "][")
     << "] = " << getName(op.getSource()) << "["
     << llvm::join(sourceIndexNames, "][") << "]\n";

  // Now unindent and close the loop nest
  for (int nestLevel = offsets.size() - 1; nestLevel >= 0; nestLevel--) {
    // Also increment the destination indices
    os << destIndexNames[nestLevel] << " += 1\n";
    os.unindent();
    os << "}\n";
  }

  // Convert to slice
  //
  // Nb., this creates a slice over the first dimension of the array, so it's a
  // slice of 1-less-dimensional arrays. This... should be fine? Because later
  // ops will use the plain indexing operator, it shouldn't matter which type
  // is used...
  os << getName(op.getResult()) << " := " << tmpName << "[:]\n";
  return success();
}

// Emits a slice copy operation and returns the string containing the emitted
// variable name of the result
std::string LattigoEmitter::emitCopySlice(Value source, Value result) {
  // Tensor semantics create new SSA values for each result. If this causes
  // inefficiencies, cf. https://github.com/google/heir/issues/1871 for ideas.
  //
  // Copy the slice:
  //
  //   result := append(make([]int, 0, len(dest)), dest...)
  //
  // Cf. https://stackoverflow.com/a/35748636
  const std::string sourceName = getName(source);
  std::string resultName = getName(result, /*force=*/true);
  os << resultName << " := append(make(" << convertType(result.getType())
     << ", 0, len(" << sourceName << ")), " << sourceName << "...)\n";
  return resultName;
}

LogicalResult LattigoEmitter::printOperation(tensor::InsertOp op) {
  const std::string resultName = emitCopySlice(op.getDest(), op.getResult());
  // result[index] = value
  os << resultName << "[";
  os << flattenIndexExpression(
      op.getResult().getType(), op.getIndices(),
      [&](Value value) { return variableNames->getNameForValue(value); });
  os << "] = " << variableNames->getNameForValue(op.getScalar()) << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::InsertSliceOp op) {
  RankedTensorType resultType = op.getResult().getType();
  if (resultType.getRank() != op.getSourceType().getRank()) {
    return op.emitError(
        "Lattigo emitter for InsertSliceOp only supports "
        "result rank equal to source rank");
  }

  // We could relax this by using previously declared SSA values for dynamic
  // offsets, sizes, and strides. But we have no use for it yet.
  if (op.getStaticOffsets().empty() || op.getStaticSizes().empty() ||
      op.getStaticStrides().empty()) {
    return op.emitError() << "expected static offsets, sizes, and strides";
  }

  const std::string destName = emitCopySlice(op.getDest(), op.getResult());
  const std::string sourceName = getName(op.getSource());

  // If we have a 1D source and target tensor, and the strides are 1,
  // we can use std::copy
  //
  // copy(dest[offset:], source);
  if (resultType.getRank() == 1 && op.getSourceType().getRank() == 1 &&
      llvm::all_of(op.getStaticStrides(),
                   [](int64_t stride) { return stride == 1; })) {
    os << "copy(" << destName << "[" << op.getStaticOffsets()[0] << ":], "
       << sourceName << ")\n";
    return success();
  }

  SmallVector<int64_t> offsets = SmallVector<int64_t>(op.getStaticOffsets());
  SmallVector<int64_t> sizes = SmallVector<int64_t>(op.getStaticSizes());
  SmallVector<int64_t> strides = SmallVector<int64_t>(op.getStaticStrides());

  // Otherwise we need a loop
  SmallVector<std::string> sourceIndexNames;
  SmallVector<std::string> destIndexNames;
  for (int nestLevel = 0; nestLevel < offsets.size(); nestLevel++) {
    std::string sourceIndexName = sourceName + "_" + std::to_string(nestLevel);
    std::string destIndexName = destName + "_" + std::to_string(nestLevel);
    sourceIndexNames.push_back(sourceIndexName);
    destIndexNames.push_back(destIndexName);
    // Initialize the source index to zero, since it is simple, note this must
    // happen outside the loop
    os << sourceIndexName << " := 0;\n";
    os << "for " << destIndexName << " := " << offsets[nestLevel] << "; "
       << destIndexName << " < "
       << offsets[nestLevel] + sizes[nestLevel] * strides[nestLevel] << "; "
       << destIndexName << " += " << strides[nestLevel] << ") {\n";
    os.indent();
  }

  // Now we're in the innermost loop nest, do the assignment
  os << destName << "[" << llvm::join(destIndexNames, "][")
     << "] = " << sourceName << "[" << llvm::join(sourceIndexNames, "][")
     << "]\n";

  // Now unindent and close the loop nest
  for (int nestLevel = offsets.size() - 1; nestLevel >= 0; nestLevel--) {
    // Also increment the source indices
    os << sourceIndexNames[nestLevel] << " += 1\n";
    os.unindent();
    os << "}\n";
  }

  return success();
}
LogicalResult LattigoEmitter::printOperation(tensor::FromElementsOp op) {
  os << getName(op.getResult()) << " := []"
     << convertType(getElementTypeOrSelf(op.getResult().getType())) << "{";
  os << getCommaSeparatedNames(op.getOperands());
  os << "}\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::SplatOp op) {
  imports.insert(std::string(kSlicesImport));
  // don't use getName because the tmpvar is guaranteed to be used
  std::string tmpVar = variableNames->getNameForValue(op.getResult()) + "_0";
  auto scalarTypeString = convertType(op.getInput().getType());

  RankedTensorType resultType = op.getResult().getType();
  if (resultType.getRank() != 1) {
    return op.emitError("Lattigo emitter for SplatOp only supports rank 1");
  }

  // golang requires a slice input to slices.Repeat
  //
  //   numbers := []int{v}
  //   repeat := slices.Repeat(numbers, 50)
  //
  int tensorSize = resultType.getNumElements();
  os << tmpVar << " := []" << scalarTypeString << "{" << getName(op.getInput())
     << "}\n";
  os << getName(op.getResult()) << " := slices.Repeat(" << tmpVar << ", "
     << tensorSize << ")\n";
  return success();
}

// RLWE

LogicalResult LattigoEmitter::printOperation(RLWENewEncryptorOp op) {
  return printNewMethod(op.getResult(), {op.getParams(), op.getEncryptionKey()},
                        "rlwe.NewEncryptor", false);
}

LogicalResult LattigoEmitter::printOperation(RLWENewDecryptorOp op) {
  return printNewMethod(op.getResult(), {op.getParams(), op.getSecretKey()},
                        "rlwe.NewDecryptor", false);
}

LogicalResult LattigoEmitter::printOperation(RLWENewKeyGeneratorOp op) {
  return printNewMethod(op.getResult(), {op.getParams()},
                        "rlwe.NewKeyGenerator", false);
}

LogicalResult LattigoEmitter::printOperation(RLWEGenKeyPairOp op) {
  return printEvalNewMethod(op.getResults(), op.getKeyGenerator(), {},
                            "GenKeyPairNew", false);
}

LogicalResult LattigoEmitter::printOperation(RLWEGenRelinearizationKeyOp op) {
  return printEvalNewMethod(op.getResult(), op.getKeyGenerator(),
                            {op.getSecretKey()}, "GenRelinearizationKeyNew",
                            false);
}

LogicalResult LattigoEmitter::printOperation(RLWEGenGaloisKeyOp op) {
  os << getName(op.getResult()) << " := " << getName(op.getKeyGenerator())
     << ".GenGaloisKeyNew(";
  os << op.getGaloisElement().getInt() << ", ";
  os << getName(op.getSecretKey()) << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(RLWENewEvaluationKeySetOp op) {
  SmallVector<Value, 4> keys;

  // verifier ensures there must be at least one key
  auto firstKey = op.getKeys()[0];
  auto galoisKeyIndex = 0;
  if (isa<RLWERelinearizationKeyType>(firstKey.getType())) {
    keys.push_back(firstKey);
    galoisKeyIndex = 1;
  } else {
    // no relinearization key, use empty Value for 'nil'
    keys.push_back(Value());
  }

  // process galois keys
  for (auto key : op.getKeys().drop_front(galoisKeyIndex)) {
    keys.push_back(key);
  }

  // EvaluationKeySet is an interface, so we need to use the concrete type
  return printNewMethod(op.getResult(), keys, "rlwe.NewMemEvaluationKeySet",
                        false);
}

LogicalResult LattigoEmitter::printOperation(RLWEEncryptOp op) {
  return printEvalNewMethod(op.getResult(), op.getEncryptor(),
                            {op.getPlaintext()}, "EncryptNew", true);
}

LogicalResult LattigoEmitter::printOperation(RLWEDecryptOp op) {
  return printEvalNewMethod(op.getResult(), op.getDecryptor(),
                            {op.getCiphertext()}, "DecryptNew", false);
}

LogicalResult LattigoEmitter::printOperation(RLWEDropLevelNewOp op) {
  // there is no DropLevelNew method in Lattigo BGV Evaluator, manually create
  // new ciphertext
  os << getName(op.getOutput()) << " := " << getName(op.getInput())
     << ".CopyNew()\n";
  os << getName(op.getEvaluator()) << ".DropLevel(" << getName(op.getOutput())
     << ", " << op.getLevelToDrop() << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(RLWEDropLevelOp op) {
  if (getName(op.getOutput()) != getName(op.getInput())) {
    os << getName(op.getOutput()) << ".Copy(" << getName(op.getInput())
       << ")\n";
  }
  os << getName(op.getEvaluator()) << ".DropLevel(" << getName(op.getOutput())
     << ", " << op.getLevelToDrop() << ")\n";
  return success();
}

namespace {
const auto* negateTemplate = R"GO(
  for {0} := 0; {0} < len({1}.Value); {0}++ {
    {2}.GetRLWEParameters().RingQ().AtLevel({1}.LevelQ()).Neg({1}.Value[{0}], {1}.Value[{0}])
  }
)GO";
}  // namespace

LogicalResult LattigoEmitter::printOperation(RLWENegateNewOp op) {
  // there is no NegateNew method in Lattigo, manually create new
  // ciphertext
  os << getName(op.getOutput()) << " := " << getName(op.getInput())
     << ".CopyNew()\n";
  auto indexName = getName(op.getOutput()) + "_index";
  auto res = llvm::formatv(negateTemplate, indexName, getName(op.getOutput()),
                           getName(op.getEvaluator()));
  os << res;
  return success();
}

LogicalResult LattigoEmitter::printOperation(RLWENegateOp op) {
  if (getName(op.getOutput()) != getName(op.getInput())) {
    os << getName(op.getOutput()) << ".Copy(" << getName(op.getInput())
       << ")\n";
  }
  auto indexName = getName(op.getOutput()) + "_index";
  auto res = llvm::formatv(negateTemplate, indexName, getName(op.getOutput()),
                           getName(op.getEvaluator()));
  os << res;
  return success();
}

// BGV

LogicalResult LattigoEmitter::printOperation(BGVNewEncoderOp op) {
  return printNewMethod(op.getResult(), {op.getParams()}, "bgv.NewEncoder",
                        false);
}

LogicalResult LattigoEmitter::printOperation(BGVNewEvaluatorOp op) {
  SmallVector<Value, 2> operands;
  operands.push_back(op.getParams());
  if (auto ekset = op.getEvaluationKeySet()) {
    operands.push_back(ekset);
  } else {
    // no evaluation key set, use empty Value for 'nil'
    operands.push_back(Value());
  }
  os << getName(op.getResult());
  os << " := bgv.NewEvaluator(";
  os << getCommaSeparatedNames(operands);
  os << ", ";
  os << (op.getScaleInvariant() ? "true" : "false");
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVNewPlaintextOp op) {
  os << getName(op.getResult()) << " := " << "bgv.NewPlaintext(";
  os << getName(op.getParams()) << ", ";
  os << getName(op.getParams()) << ".MaxLevel()";
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVEncodeOp op) {
  // cyclic repetition to mitigate openfhe zero-padding (#645)
  // TODO(#1258): move cyclic repetition to earlier pipeline

  // hack: access another op to get params then get MaxSlots
  auto newPlaintextOp =
      mlir::dyn_cast<BGVNewPlaintextOp>(op.getPlaintext().getDefiningOp());
  if (!newPlaintextOp) {
    return failure();
  }
  auto maxSlotsName = getName(newPlaintextOp.getParams()) + ".MaxSlots()";

  // EncodeOp requires its argument to be a slice of int64
  // so besides cyclic full packing behavior, we are also doing type conversion
  auto packedName =
      getName(op.getValue()) + "_" + getName(op.getPlaintext()) + "_packed";
  os << packedName << " := make([]int64, ";
  os << maxSlotsName << ")\n";
  os << "for i := range " << packedName << " {\n";
  auto valueNameAtI =
      getName(op.getValue()) + "[i % len(" + getName(op.getValue()) + ")]";
  auto packedNameAtI = packedName + "[i]";
  os.indent();
  if (getElementTypeOrSelf(op.getValue().getType()).getIntOrFloatBitWidth() ==
      1) {
    emitIf(
        valueNameAtI, [&]() { os << packedNameAtI << " = int64(1)\n"; },
        [&]() { os << packedNameAtI << " = int64(0)\n"; });
  } else {
    // packedName[i] = int64(value[i % len(value)])
    os << packedNameAtI << " = int64(" << valueNameAtI << ")\n";
  }
  os.unindent();
  os << "}\n";

  // set the scale of plaintext
  auto scale = op.getScale();
  os << getName(op.getPlaintext()) << ".Scale = ";
  os << getName(newPlaintextOp.getParams()) << ".NewScale(";
  os << scale << ")\n";

  os << getName(op.getEncoder()) << ".Encode(";
  os << packedName << ", ";
  os << getName(op.getPlaintext()) << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVDecodeOp op) {
  // Value itself may be []int16, so make a []int64 slice for the decoded value,
  // as Decode only accepts []int64
  auto valueInt64Name = getName(op.getValue()) + "_int64";
  os << valueInt64Name << " := make([]int64, " << "len("
     << getName(op.getValue()) << "))\n";

  os << getName(op.getEncoder()) << ".Decode(";
  os << getName(op.getPlaintext()) << ", ";
  os << valueInt64Name << ")\n";

  // type conversion from value to decoded
  auto convertedName = getName(op.getDecoded()) + "_converted";
  os << convertedName << " := make(" << convertType(op.getDecoded().getType())
     << ", len(" << getName(op.getValue()) << "))\n";
  os << "for i := range " << getName(op.getValue()) << " {\n";
  os.indent();
  os << convertedName
     << "[i] = " << convertType(getElementTypeOrSelf(op.getDecoded().getType()))
     << "(" << valueInt64Name << "[i])\n";
  os.unindent();
  os << "}\n";
  os << getName(op.getDecoded()) << " := " << convertedName << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVAddNewOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "AddNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVSubNewOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "SubNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVMulNewOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "MulNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVAddOp op) {
  return printEvalInplaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Add", true);
}

LogicalResult LattigoEmitter::printOperation(BGVSubOp op) {
  return printEvalInplaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Sub", true);
}

LogicalResult LattigoEmitter::printOperation(BGVMulOp op) {
  return printEvalInplaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Mul", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRelinearizeNewOp op) {
  return printEvalNewMethod(op.getOutput(), op.getEvaluator(), op.getInput(),
                            "RelinearizeNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRescaleNewOp op) {
  // there is no RescaleNew method in Lattigo, manually create new ciphertext
  os << getName(op.getOutput()) << " := " << getName(op.getInput())
     << ".CopyNew()\n";
  return printEvalInplaceMethod(
      op.getEvaluator(), {op.getInput(), op.getOutput()}, "Rescale", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRotateColumnsNewOp op) {
  auto errName = getErrName();
  os << getName(op.getOutput()) << ", " << errName
     << " := " << getName(op.getEvaluator()) << ".RotateColumnsNew(";
  os << getName(op.getInput()) << ", ";
  os << op.getOffset().getInt() << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVRotateRowsNewOp op) {
  return printEvalNewMethod(op.getOutput(), op.getEvaluator(), {op.getInput()},
                            "RotateRowsNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRelinearizeOp op) {
  return printEvalInplaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "Relinearize", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRescaleOp op) {
  return printEvalInplaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "Rescale", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRotateColumnsOp op) {
  auto errName = getErrName();
  os << errName << " := " << getName(op.getEvaluator()) << ".RotateColumns(";
  os << getName(op.getInput()) << ", ";
  os << op.getOffset().getInt() << ", ";
  os << getName(op.getInplace()) << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVRotateRowsOp op) {
  return printEvalInplaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "RotateRows", true);
}

std::string printDenseI32ArrayAttr(DenseI32ArrayAttr attr) {
  std::string res = "[]int{";
  res += commaSeparated(attr.asArrayRef());
  res += "}";
  return res;
}

// use i64 for u64 now
std::string printDenseU64ArrayAttr(DenseI64ArrayAttr attr) {
  std::string res = "[]uint64{";
  res += commaSeparated(attr.asArrayRef());
  res += "}";
  return res;
}

LogicalResult LattigoEmitter::printOperation(BGVNewParametersFromLiteralOp op) {
  auto errName = getErrName();
  os << getName(op.getResult()) << ", " << errName
     << " := bgv.NewParametersFromLiteral(";
  os << "bgv.ParametersLiteral{\n";
  os.indent();
  os << "LogN: " << op.getParamsLiteral().getLogN() << ",\n";
  if (auto Q = op.getParamsLiteral().getQ()) {
    os << "Q: " << printDenseU64ArrayAttr(Q) << ",\n";
  }
  if (auto P = op.getParamsLiteral().getP()) {
    os << "P: " << printDenseU64ArrayAttr(P) << ",\n";
  }
  if (auto LogQ = op.getParamsLiteral().getLogQ()) {
    os << "LogQ: " << printDenseI32ArrayAttr(LogQ) << ",\n";
  }
  if (auto LogP = op.getParamsLiteral().getLogP()) {
    os << "LogP: " << printDenseI32ArrayAttr(LogP) << ",\n";
  }
  os << "PlaintextModulus: " << op.getParamsLiteral().getPlaintextModulus()
     << ",\n";
  os.unindent();
  os << "})\n";
  printErrPanic(errName);
  return success();
}

// CKKS

LogicalResult LattigoEmitter::printOperation(CKKSNewEncoderOp op) {
  return printNewMethod(op.getResult(), {op.getParams()}, "ckks.NewEncoder",
                        false);
}

LogicalResult LattigoEmitter::printOperation(CKKSNewEvaluatorOp op) {
  SmallVector<Value, 2> operands;
  operands.push_back(op.getParams());
  if (auto ekset = op.getEvaluationKeySet()) {
    operands.push_back(ekset);
  } else {
    // no evaluation key set, use empty Value for 'nil'
    operands.push_back(Value());
  }
  return printNewMethod(op.getResult(), operands, "ckks.NewEvaluator", false);
}

LogicalResult LattigoEmitter::printOperation(CKKSNewPlaintextOp op) {
  os << getName(op.getResult()) << " := " << "ckks.NewPlaintext(";
  os << getName(op.getParams()) << ", ";
  os << getName(op.getParams()) << ".MaxLevel()";
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSEncodeOp op) {
  // cyclic repetition to mitigate openfhe zero-padding (#645)
  // TODO(#1258): move cyclic repetition to earlier pipeline

  // hack: access another op to get params then get MaxSlots
  auto newPlaintextOp =
      mlir::dyn_cast<CKKSNewPlaintextOp>(op.getPlaintext().getDefiningOp());
  if (!newPlaintextOp) {
    return failure();
  }
  auto maxSlotsName = getName(newPlaintextOp.getParams()) + ".MaxSlots()";

  // EncodeOp requires its argument to be a slice of float64
  // so besides cyclic full packing behavior, we are also doing type conversion
  auto packedName =
      getName(op.getValue()) + "_" + getName(op.getPlaintext()) + "_packed";
  os << packedName << " := make([]float64, ";
  os << maxSlotsName << ")\n";
  os << "for i := range " << packedName << " {\n";
  auto valueNameAtI =
      getName(op.getValue()) + "[i % len(" + getName(op.getValue()) + ")]";
  auto packedNameAtI = packedName + "[i]";
  os.indent();
  if (getElementTypeOrSelf(op.getValue().getType()).getIntOrFloatBitWidth() ==
      1) {
    const auto* boolToFloat64Template = R"GO(
      if {0} {
        {1} = 1.0
      } else {
        {1} = 0.0
      }
    )GO";
    auto res =
        llvm::formatv(boolToFloat64Template, valueNameAtI, packedNameAtI);
    os << res;
  } else {
    // packedName[i] = float64(value[i % len(value)])
    os << packedNameAtI << " = float64(" << valueNameAtI << ")\n";
  }
  os.unindent();
  os << "}\n";

  // set the scale of plaintext
  auto scale = op.getScale();
  os << getName(op.getPlaintext()) << ".Scale = ";
  os << getName(newPlaintextOp.getParams()) << ".NewScale(math.Pow(2, ";
  os << scale << "))\n";

  os << getName(op.getEncoder()) << ".Encode(";
  os << packedName << ", ";
  os << getName(op.getPlaintext()) << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSDecodeOp op) {
  // Value itself may be []float32, so make a []float64 slice for the decoded
  // value, as Decode only accepts []float64
  auto valueFloat64Name = getName(op.getValue()) + "_float64";
  os << valueFloat64Name << " := make([]float64, " << "len("
     << getName(op.getValue()) << "))\n";

  os << getName(op.getEncoder()) << ".Decode(";
  os << getName(op.getPlaintext()) << ", ";
  os << valueFloat64Name << ")\n";

  // type conversion from value to decoded
  auto convertedName = getName(op.getDecoded()) + "_converted";
  os << convertedName << " := make(" << convertType(op.getDecoded().getType())
     << ", len(" << getName(op.getValue()) << "))\n";
  os << "for i := range " << getName(op.getValue()) << " {\n";
  os.indent();
  os << convertedName
     << "[i] = " << convertType(getElementTypeOrSelf(op.getDecoded().getType()))
     << "(" << valueFloat64Name << "[i])\n";
  os.unindent();
  os << "}\n";
  os << getName(op.getDecoded()) << " := " << convertedName << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSAddNewOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "AddNew", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSSubNewOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "SubNew", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSMulNewOp op) {
  return printEvalNewMethod(op.getResult(), op.getEvaluator(),
                            {op.getLhs(), op.getRhs()}, "MulNew", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSAddOp op) {
  return printEvalInplaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Add", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSSubOp op) {
  return printEvalInplaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Sub", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSMulOp op) {
  return printEvalInplaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Mul", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRelinearizeNewOp op) {
  return printEvalNewMethod(op.getOutput(), op.getEvaluator(), op.getInput(),
                            "RelinearizeNew", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRescaleNewOp op) {
  // there is no RescaleNew method in Lattigo, manually create new ciphertext
  os << getName(op.getOutput()) << " := " << getName(op.getInput())
     << ".CopyNew()\n";
  return printEvalInplaceMethod(
      op.getEvaluator(), {op.getInput(), op.getOutput()}, "Rescale", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRotateNewOp op) {
  auto errName = getErrName();
  os << getName(op.getOutput()) << ", " << errName
     << " := " << getName(op.getEvaluator()) << ".RotateNew(";
  os << getName(op.getInput()) << ", ";
  os << op.getOffset().getInt() << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSRelinearizeOp op) {
  return printEvalInplaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "Relinearize", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRescaleOp op) {
  return printEvalInplaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "Rescale", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRotateOp op) {
  auto errName = getErrName();
  os << errName << " := " << getName(op.getEvaluator()) << ".Rotate(";
  os << getName(op.getInput()) << ", ";
  os << op.getOffset().getInt() << ", ";
  os << getName(op.getInplace()) << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(
    CKKSNewParametersFromLiteralOp op) {
  auto errName = getErrName();
  os << getName(op.getResult()) << ", " << errName
     << " := ckks.NewParametersFromLiteral(";
  os << "ckks.ParametersLiteral{\n";
  os.indent();
  os << "LogN: " << op.getParamsLiteral().getLogN() << ",\n";
  if (auto Q = op.getParamsLiteral().getQ()) {
    os << "Q: " << printDenseU64ArrayAttr(Q) << ",\n";
  }
  if (auto P = op.getParamsLiteral().getP()) {
    os << "P: " << printDenseU64ArrayAttr(P) << ",\n";
  }
  if (auto LogQ = op.getParamsLiteral().getLogQ()) {
    os << "LogQ: " << printDenseI32ArrayAttr(LogQ) << ",\n";
  }
  if (auto LogP = op.getParamsLiteral().getLogP()) {
    os << "LogP: " << printDenseI32ArrayAttr(LogP) << ",\n";
  }
  os << "LogDefaultScale: " << op.getParamsLiteral().getLogDefaultScale()
     << ",\n";
  os.unindent();
  os << "})\n";
  printErrPanic(errName);
  return success();
}

void LattigoEmitter::printErrPanic(std::string_view errName) {
  os << "if " << errName << " != nil {\n";
  os.indent();
  os << "panic(" << errName << ")\n";
  os.unindent();
  os << "}\n";
}

LogicalResult LattigoEmitter::printNewMethod(::mlir::Value result,
                                             ::mlir::ValueRange operands,
                                             std::string_view op, bool err) {
  std::string errName = getErrName();
  os << getName(result);
  if (err) {
    os << ", " << errName;
  }
  os << " := " << op << "(";
  os << getCommaSeparatedNames(operands);
  os << ")\n";
  if (err) {
    printErrPanic(errName);
  }
  return success();
}

LogicalResult LattigoEmitter::printEvalInplaceMethod(
    ::mlir::Value evaluator, ::mlir::ValueRange operands, std::string_view op,
    bool err) {
  std::string errName = getErrName();
  if (err) {
    os << errName << " := ";
  }
  os << getName(evaluator) << "." << op << "("
     << getCommaSeparatedNames(operands) << ");\n";
  if (err) {
    printErrPanic(errName);
  }
  return success();
}

LogicalResult LattigoEmitter::printEvalNewMethod(::mlir::ValueRange results,
                                                 ::mlir::Value evaluator,
                                                 ::mlir::ValueRange operands,
                                                 std::string_view op,
                                                 bool err) {
  std::string errName = getErrName();
  os << getCommaSeparatedNames(results);
  if (err) {
    os << ", " << errName;
  }
  os << " := " << getName(evaluator) << "." << op << "(";
  os << getCommaSeparatedNames(operands);
  os << ")\n";
  if (err) {
    printErrPanic(errName);
  }
  return success();
}

FailureOr<std::string> LattigoEmitter::convertType(Type type) {
  return llvm::TypeSwitch<Type, FailureOr<std::string>>(type)
      // RLWE
      .Case<RLWECiphertextType>(
          [&](auto ty) { return std::string("*rlwe.Ciphertext"); })
      .Case<RLWEPlaintextType>(
          [&](auto ty) { return std::string("*rlwe.Plaintext"); })
      .Case<RLWESecretKeyType>(
          [&](auto ty) { return std::string("*rlwe.PrivateKey"); })
      .Case<RLWEPublicKeyType>(
          [&](auto ty) { return std::string("*rlwe.PublicKey"); })
      .Case<RLWEKeyGeneratorType>(
          [&](auto ty) { return std::string("*rlwe.KeyGenerator"); })
      .Case<RLWERelinearizationKeyType>(
          [&](auto ty) { return std::string("*rlwe.RelinearizationKey"); })
      .Case<RLWEGaloisKeyType>(
          [&](auto ty) { return std::string("*rlwe.GaloisKey"); })
      .Case<RLWEEvaluationKeySetType>(
          [&](auto ty) { return std::string("*rlwe.EvaluationKeySet"); })
      .Case<RLWEEncryptorType>(
          [&](auto ty) { return std::string("*rlwe.Encryptor"); })
      .Case<RLWEDecryptorType>(
          [&](auto ty) { return std::string("*rlwe.Decryptor"); })
      // BGV
      .Case<BGVEncoderType>(
          [&](auto ty) { return std::string("*bgv.Encoder"); })
      .Case<BGVEvaluatorType>(
          [&](auto ty) { return std::string("*bgv.Evaluator"); })
      .Case<BGVParameterType>(
          [&](auto ty) { return std::string("bgv.Parameters"); })
      // CKKS
      .Case<CKKSEncoderType>(
          [&](auto ty) { return std::string("*ckks.Encoder"); })
      .Case<CKKSEvaluatorType>(
          [&](auto ty) { return std::string("*ckks.Evaluator"); })
      .Case<CKKSParameterType>(
          [&](auto ty) { return std::string("ckks.Parameters"); })
      .Case<IntegerType>([&](auto ty) -> FailureOr<std::string> {
        auto width = ty.getWidth();
        if (width == 1) {
          return std::string("bool");
        }
        if (width != 8 && width != 16 && width != 32 && width != 64) {
          return failure();
        }
        return "int" + std::to_string(width);
      })
      .Case<FloatType>([&](auto ty) -> FailureOr<std::string> {
        auto width = ty.getWidth();
        if (width == 16 || width == 8) {
          width = 32;
          // emitWarning(loc, "Floating point width " + std::to_string(width) +
          //             " is not supported in GO, using 32-bit float
          //             instead.");
        }
        if (width != 32 && width != 64) {
          return failure();
        }
        return "float" + std::to_string(width);
      })
      .Case<IndexType>([&](auto ty) -> FailureOr<std::string> {
        // Translate IndexType to int64 in GO
        return std::string("int64");
      })
      .Case<RankedTensorType>([&](auto ty) -> FailureOr<std::string> {
        auto eltTyResult = convertType(ty.getElementType());
        if (failed(eltTyResult)) {
          return failure();
        }
        auto result = eltTyResult.value();
        return std::string("[]") + result;
      })
      .Default([&](Type) -> FailureOr<std::string> { return failure(); });
}

LogicalResult LattigoEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

LattigoEmitter::LattigoEmitter(raw_ostream& os,
                               SelectVariableNames* variableNames,
                               const std::string& packageName)
    : os(os), variableNames(variableNames), packageName(packageName) {}

struct TranslateOptions {
  llvm::cl::opt<std::string> packageName{
      "package-name",
      llvm::cl::desc("The name to use for the package declaration in the "
                     "generated golang file.")};
};
static llvm::ManagedStatic<TranslateOptions> translateOptions;

void registerTranslateOptions() {
  // Forces initialization of options.
  *translateOptions;
}

void registerToLattigoTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-lattigo",
      "translate the lattigo dialect to GO code against the Lattigo API",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToLattigo(op, output, translateOptions->packageName);
      },
      [](DialectRegistry& registry) {
        registry.insert<affine::AffineDialect, rns::RNSDialect,
                        arith::ArithDialect, func::FuncDialect,
                        tensor::TensorDialect, tensor_ext::TensorExtDialect,
                        lattigo::LattigoDialect, mgmt::MgmtDialect>();
      });
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
