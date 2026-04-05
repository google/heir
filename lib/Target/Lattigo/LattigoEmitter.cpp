#include "lib/Target/Lattigo/LattigoEmitter.h"

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

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
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StaticValueUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"           // from @llvm-project
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
                                 const std::string& packageName,
                                 const std::vector<std::string>& extraImports,
                                 std::function<bool(func::FuncOp)> funcFilter) {
  SelectVariableNames variableNames(op);
  std::string bufferedStr;
  llvm::raw_string_ostream strOs(bufferedStr);
  raw_indented_ostream bufferedOs(strOs);
  LattigoEmitter emitter(bufferedOs, &variableNames, packageName, extraImports,
                         funcFilter);
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
                arith::FloorDivSIOp, arith::IndexCastOp, arith::ExtFOp,
                arith::RemSIOp, arith::AddIOp, arith::AddFOp, arith::AndIOp,
                arith::SubIOp, arith::SubFOp, arith::MaxSIOp, arith::MinSIOp,
                arith::MulIOp, arith::MulFOp, arith::DivSIOp, arith::DivFOp,
                arith::NegFOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp>(
              [&](auto op) { return printOperation(op); })
          // SCF ops
          .Case<scf::IfOp, scf::ForOp, scf::YieldOp>(
              [&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ConcatOp, tensor::EmptyOp, tensor::ExtractOp,
                tensor::ExtractSliceOp, tensor::InsertOp, tensor::InsertSliceOp,
                tensor::FromElementsOp, tensor::SplatOp, tensor::ExpandShapeOp,
                tensor::CollapseShapeOp>(
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
              CKKSNewEvaluatorOp, CKKSNewPlaintextOp,
              CKKSNewPolynomialEvaluatorOp, CKKSEncodeOp, CKKSDecodeOp,
              CKKSAddNewOp, CKKSSubNewOp, CKKSMulNewOp, CKKSAddOp, CKKSSubOp,
              CKKSMulOp, CKKSRelinearizeOp, CKKSRescaleOp, CKKSRotateOp,
              CKKSRelinearizeNewOp, CKKSRescaleNewOp, CKKSRotateNewOp,
              CKKSLinearTransformOp, CKKSChebyshevOp, CKKSBootstrapOp,
              CKKSNewBootstrappingParametersFromLiteralOp,
              CKKSGenEvaluationKeysBootstrappingOp,
              CKKSNewBootstrappingEvaluatorOp>(
              [&](auto op) { return printOperation(op); })
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
    imports.insert(std::string(kCkksImport));
  } else {
    return moduleOp.emitError("Unknown scheme");
  }

  for (const auto& extraImport : extraImports) {
    imports.insert("\"" + extraImport + "\"");
  }

  for (Operation& op : moduleOp) {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (funcFilter && !funcFilter(funcOp)) {
        continue;
      }
    }
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
  std::string funcName = funcOp.getName().str();
  if (funcOp->hasAttr(kClientPackFuncAttrName)) {
    if (!funcName.empty() && funcFilter && funcFilter(funcOp) &&
        std::islower(funcName[0])) {
      // Upper case this only if we are emitting it in a filtered file so it
      // needs exporting.
      funcName[0] = std::toupper(funcName[0]);
    }
  }
  os << "func " << funcName << "(";
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

  auto callee = op.getCallee();
  auto moduleOp = op->getParentOfType<ModuleOp>();
  auto calleeOp = moduleOp.lookupSymbol<func::FuncOp>(callee);
  std::string calleeName = canonicalizeDebugPort(callee).str();
  if (calleeOp && funcFilter && !funcFilter(calleeOp)) {
    if (calleeOp->hasAttr(kClientPackFuncAttrName)) {
      if (!calleeName.empty() && std::islower(calleeName[0])) {
        calleeName[0] = std::toupper(calleeName[0]);
      }
      calleeName = packageName + "_utils." + calleeName;
    }
  }

  if (op.getNumResults() > 0) {
    os << getCommaSeparatedNames(op.getResults());
    os << " := ";
  }
  os << calleeName << "(";
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
    Value result = op.getResults()[i];
    Value init = op.getOperands()[i];
    Value iterArg = op.getRegionIterArgs()[i];

    if (variableNames->contains(result) && variableNames->contains(init) &&
        variableNames->getNameForValue(result) ==
            variableNames->getNameForValue(init)) {
      // This occurs in cases where the loop is inserting into a tensor and
      // passing it along as an inter arg.
      continue;
    }

    os << getName(iterArg) << " := " << getName(init) << "\n";
  }

  os << llvm::formatv("for {0} := {1}; {0} < {2}; {0} += {3} {{\n",
                      variableNames->getNameForValue(op.getInductionVar()),
                      op.getConstantLowerBound(), op.getConstantUpperBound(),
                      op.getStepAsInt());
  os.indent();
  for (Operation& op : *op.getBody()) {
    if (failed(translate(op))) {
      return op.emitOpError() << "Failed to translate for loop block";
    }
  }

  os.unindent();
  os << "}\n";

  // Finally, forward the iter_arg to the op results.
  for (auto i = 0; i < op.getNumRegionIterArgs(); ++i) {
    Value opResult = op.getResult(i);
    Value iterArg = op.getRegionIterArgs()[i];
    os << getName(opResult) << " := " << getName(iterArg) << "\n";
  }
  return success();
}

LogicalResult LattigoEmitter::printOperation(affine::AffineYieldOp op) {
  FailureOr<ValueRange> destValues =
      llvm::TypeSwitch<Operation*, FailureOr<ValueRange>>(op->getParentOp())
          .Case<affine::AffineForOp>(
              [&](auto forOp) { return forOp.getRegionIterArgs(); })
          // TODO: support AffineIfOp
          .Default([&](auto op) { return failure(); });

  if (failed(destValues)) {
    return failure();
  }

  for (int i = 0; i < op.getNumOperands(); ++i) {
    Value dest = destValues.value()[i];
    Value yieldedValue = op.getOperand(i);
    os << getName(dest) << " = " << getName(yieldedValue) << "\n";
  }

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

FailureOr<std::string> getStringForElementsAttr(ElementsAttr elementsAttr) {
  // Splat is also handled in getValues
  SmallVector<std::string> values;
  Type elementType = elementsAttr.getElementType();
  if (mlir::isa<IntegerType, IndexType>(elementType)) {
    for (auto value : elementsAttr.getValues<APInt>()) {
      auto constantString = getStringForConstant(value);
      if (failed(constantString)) {
        return failure();
      }
      values.push_back(*constantString);
    }
  } else if (mlir::isa<FloatType>(elementType)) {
    for (auto value : elementsAttr.getValues<APFloat>()) {
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

bool isConstantOne(OpFoldResult ofr) {
  auto val = mlir::getConstantIntValue(ofr);
  return val && *val == 1;
}

}  // namespace

LogicalResult LattigoEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  auto type = valueAttr.getType();

  if (auto resourceAttr =
          mlir::dyn_cast<DenseResourceElementsAttr>(valueAttr)) {
    valueAttr = DenseElementsAttr::getFromRawBuffer(resourceAttr.getType(),
                                                    resourceAttr.getData());
  }

  auto typeString = convertType(type);
  if (failed(typeString)) {
    return failure();
  }

  if (auto elementsAttr = mlir::dyn_cast<ElementsAttr>(valueAttr)) {
    int64_t numElements = mlir::cast<RankedTensorType>(type).getNumElements();
    if (elementsAttr.isSplat()) {
      Attribute splatValue = elementsAttr.getSplatValue<Attribute>();
      bool isZero = false;
      if (auto intAttr = mlir::dyn_cast<IntegerAttr>(splatValue)) {
        isZero = intAttr.getValue().isZero();
      } else if (auto floatAttr = mlir::dyn_cast<FloatAttr>(splatValue)) {
        isZero = floatAttr.getValue().isZero();
      }

      if (isZero) {
        os << getName(op.getResult()) << " := make(" << *typeString << ", "
           << numElements << ")\n";
        return success();
      }

      if (numElements > 8) {
        imports.insert(std::string(kSlicesImport));
        auto eltTypeString =
            convertType(mlir::cast<RankedTensorType>(type).getElementType());
        if (failed(eltTypeString)) return failure();

        std::string valStr;
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(splatValue)) {
          valStr = getStringForConstant(intAttr.getValue()).value();
        } else if (auto floatAttr = mlir::dyn_cast<FloatAttr>(splatValue)) {
          valStr = getStringForConstant(floatAttr.getValue()).value();
        }

        os << getName(op.getResult()) << " := slices.Repeat([]"
           << *eltTypeString << "{" << valStr << "}, " << numElements << ")\n";
        return success();
      }
    }
  }

  // GO use () for scalar: int64()
  std::string left = "(";
  std::string right = ")";
  if (mlir::isa<RankedTensorType>(type)) {
    // GO use {} for slice: []int64{}
    left = "{";
    right = "}";
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
          .Case<ElementsAttr>([&](ElementsAttr elementsAttr) {
            // fill in the values
            auto constString = getStringForElementsAttr(elementsAttr);
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

LogicalResult LattigoEmitter::typecast(Value operand, Value result,
                                       bool isSigned) {
  std::string inputVarName = getName(operand);
  Type inputEltTy = getElementTypeOrSelf(operand.getType());
  Type resultEltTy = getElementTypeOrSelf(result.getType());
  bool isBoolInput = inputEltTy.isInteger(1);
  bool isBoolResult = resultEltTy.isInteger(1);

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

    if (isBoolInput) {
      emitIf(
          "val",
          [&]() {
            os << resultVarName << "[i] = " << elementType << "("
               << (isSigned ? "-1" : "1") << ")\n";
          },
          [&]() { os << resultVarName << "[i] = " << elementType << "(0)\n"; });
    } else if (isBoolResult) {
      os << resultVarName << "[i] = val != 0\n";
    } else {
      os << resultVarName << "[i] = " << elementType << "(val)\n";
    }

    os.unindent();
    os << "}\n";
  } else {
    auto res = convertType(result.getType());
    if (failed(res)) {
      return failure();
    }
    std::string resultTy = res.value();
    if (isBoolInput) {
      std::string resultName =
          declareVariable(result, result.getDefiningOp()->getLoc());
      emitIf(
          inputVarName,
          [&]() {
            os << resultName << " = " << resultTy << "("
               << (isSigned ? "-1" : "1") << ")\n";
          },
          [&]() { os << resultName << " = " << resultTy << "(0)\n"; });
    } else if (isBoolResult) {
      os << getName(result) << " := " << inputVarName << " != 0\n";
    } else {
      os << getName(result) << " := " << resultTy << "(" << inputVarName
         << ")\n";
    }
  }

  return success();
}

LogicalResult LattigoEmitter::printOperation(arith::ExtSIOp op) {
  return typecast(op.getOperand(), op.getResult(), /*isSigned=*/true);
}

LogicalResult LattigoEmitter::printOperation(arith::ExtUIOp op) {
  return typecast(op.getOperand(), op.getResult(), /*isSigned=*/false);
}

LogicalResult LattigoEmitter::printOperation(arith::ExtFOp op) {
  return typecast(op.getOperand(), op.getResult());
}

LogicalResult LattigoEmitter::printOperation(arith::FloorDivSIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "/");
}

LogicalResult LattigoEmitter::printOperation(arith::IndexCastOp op) {
  return typecast(op.getOperand(), op.getOut());
}

LogicalResult LattigoEmitter::printBinaryOp(Operation* op, ::mlir::Value lhs,
                                            ::mlir::Value rhs,
                                            std::string_view opName) {
  assert(op->getNumResults() == 1 && "Expected single-result op!");
  Value result = op->getResult(0);
  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    std::string resultName = getName(result);
    auto res = convertType(tensorTy);
    if (failed(res)) return failure();

    os << resultName << " := make(" << res.value() << ", "
       << tensorTy.getNumElements() << ")\n";
    os << "for i := range " << resultName << " {\n";
    os.indent();
    os << resultName << "[i] = " << getName(lhs) << "[i] " << opName << " "
       << getName(rhs) << "[i]\n";
    os.unindent();
    os << "}\n";
    return success();
  }

  os << getName(result) << " := " << getName(lhs) << " " << opName << " "
     << getName(rhs) << "\n";
  return success();
}

// Lowerings of ops like affine.apply involve scalar cleartext types
LogicalResult LattigoEmitter::printOperation(arith::AddIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "+");
}

LogicalResult LattigoEmitter::printOperation(arith::AddFOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "+");
}

LogicalResult LattigoEmitter::printOperation(arith::AndIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "&");
}

LogicalResult LattigoEmitter::printFunctionalBinop(Operation* op,
                                                   ::mlir::Value lhs,
                                                   ::mlir::Value rhs,
                                                   std::string_view opName) {
  Value result = op->getResults()[0];
  std::string resultName = getName(result);
  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    auto res = convertType(tensorTy);
    if (failed(res)) return failure();

    os << resultName << " := make(" << res.value() << ", "
       << tensorTy.getNumElements() << ")\n";
    os << "for i := range " << resultName << " {\n";
    os.indent();
    os << resultName << "[i] = " << opName << "(" << getName(lhs) << "[i], "
       << getName(rhs) << "[i])\n";
    os.unindent();
    os << "}\n";
    return success();
  }

  os << resultName << " := " << opName << "(" << getName(lhs) << ", "
     << getName(rhs) << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(arith::MaxSIOp op) {
  return printFunctionalBinop(op, op.getLhs(), op.getRhs(), "max");
}

LogicalResult LattigoEmitter::printOperation(arith::MinSIOp op) {
  return printFunctionalBinop(op, op.getLhs(), op.getRhs(), "min");
}

LogicalResult LattigoEmitter::printOperation(arith::MulIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "*");
}

LogicalResult LattigoEmitter::printOperation(arith::MulFOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "*");
}

LogicalResult LattigoEmitter::printOperation(arith::SubIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "-");
}

LogicalResult LattigoEmitter::printOperation(arith::SubFOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "-");
}

LogicalResult LattigoEmitter::printOperation(arith::DivSIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "/");
}

LogicalResult LattigoEmitter::printOperation(arith::DivFOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "/");
}

LogicalResult LattigoEmitter::printOperation(arith::NegFOp op) {
  os << getName(op.getResult()) << " := -" << getName(op.getOperand()) << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(arith::RemSIOp op) {
  return printBinaryOp(op, op.getLhs(), op.getRhs(), "%");
}

LogicalResult LattigoEmitter::printOperation(arith::SelectOp op) {
  std::string resultName = declareVariable(op.getResult(), op.getLoc());
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

LogicalResult LattigoEmitter::printOperation(arith::CmpFOp op) {
  std::string resultName = getName(op.getResult());
  switch (op.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse:
      os << resultName << " = false\n";
      return success();
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), "==");
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), ">");
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), ">=");
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), "<");
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), "<=");
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      return printBinaryOp(op, op.getLhs(), op.getRhs(), "!=");
    case arith::CmpFPredicate::AlwaysTrue:
      os << resultName << " true\n";
      return success();
    default:
      return op.emitOpError("unsupported cmpf predicate");
  }
}

LogicalResult LattigoEmitter::printOperation(scf::IfOp op) {
  // var result <type>
  // if cond {
  //   result = blah
  // } else {
  //   result = blah
  // }
  for (auto i = 0; i < op.getNumResults(); ++i) {
    Value result = op.getResults()[i];
    declareVariable(result, op.getLoc());
  }

  os << llvm::formatv("if {0} {{\n",
                      variableNames->getNameForValue(op.getCondition()));
  os.indent();
  for (Operation& op : *op.thenBlock()) {
    if (failed(translate(op))) {
      return op.emitOpError() << "Failed to translate for if then block";
    }
  }
  os.unindent();
  os << "}";

  os << llvm::formatv(" else {{\n");
  os.indent();
  for (Operation& op : *op.elseBlock()) {
    if (failed(translate(op))) {
      return op.emitOpError() << "Failed to translate for if else block";
    }
  }
  os.unindent();
  os << "}";

  os << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(scf::ForOp op) {
  for (auto i = 0; i < op.getNumRegionIterArgs(); ++i) {
    Value result = op.getResults()[i];
    Value init = op.getInitArgs()[i];
    Value iterArg = op.getRegionIterArgs()[i];

    if (variableNames->contains(result) && variableNames->contains(init) &&
        getName(result) == getName(init)) {
      // This occurs in cases where the loop is inserting into a tensor and
      // passing it along as an inter arg.
      continue;
    }

    // Initialize each iter_arg to its corresponding init
    os << getName(iterArg) << " := " << getName(init) << "\n";
  }

  auto getConstantOrValue = [&](Value value) -> std::string {
    return getStringForConstant(value).value_or(getName(value));
  };

  os << llvm::formatv("for {0} := {1}; {0} < {2}; {0} += {3} {{\n",
                      variableNames->getNameForValue(op.getInductionVar()),
                      getConstantOrValue(op.getLowerBound()),
                      getConstantOrValue(op.getUpperBound()),
                      getConstantOrValue(op.getStep()));
  os.indent();
  for (Operation& op : *op.getBody()) {
    if (failed(translate(op))) {
      return op.emitOpError() << "Failed to translate for loop block";
    }
  }

  os.unindent();
  os << "}\n";

  // Finally, forward the iter_arg to the op results.
  for (int i = 0; i < op.getNumResults(); ++i) {
    Value opResult = op.getResult(i);
    Value iterArg = op.getRegionIterArg(i);
    os << getName(opResult) << " := " << getName(iterArg) << "\n";
  }
  return success();
}

LogicalResult LattigoEmitter::printOperation(scf::YieldOp op) {
  // Yield is used in both if statements (which always have a single successor
  // region (the parent op), and loops (which have multiple successor regions).
  // In the case of an if statement, we want to assign to the parent successor
  // values, and in the case of a for loop, we want to assign to the iter_args
  // and let the parent op handle the forwarding of iter_args to the final op
  // results.
  //
  // I don't see an elegant way to do that without just type-casing on each
  // parent op type.
  ValueRange destValues(
      llvm::TypeSwitch<Operation*, ValueRange>(op->getParentOp())
          .Case<scf::ForOp>(
              [&](auto forOp) { return forOp.getRegionIterArgs(); })
          .Case<scf::IfOp>([&](auto ifOp) { return ifOp.getResults(); })
          .Default([&](auto op) { return ValueRange{}; }));

  for (int i = 0; i < op.getNumOperands(); ++i) {
    Value dest = destValues[i];
    Value yieldedValue = op.getOperand(i);
    os << getName(dest) << " = " << getName(yieldedValue) << "\n";
  }

  return success();
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
  // Tensors are flattened into 1D slices.
  ShapedType resultType = op.getResult().getType();
  auto typeStr = convertType(resultType.getElementType());
  std::string resultName = getName(op.getResult());
  os << getName(op.getResult()) << " := make([]" << typeStr.value() << ", "
     << resultType.getNumElements() << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::ExtractOp op) {
  std::string resultName = getName(op.getResult());
  os << getName(op.getResult()) << " := " << getName(op.getTensor()) << "[";
  os << flattenIndexExpression(
      op.getTensor().getType(), op.getIndices(),
      [&](Value value) { return variableNames->getNameForValue(value); });
  os << "]\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::ExtractSliceOp op) {
  RankedTensorType resultType = op.getResult().getType();
  RankedTensorType sourceType = op.getSource().getType();

  std::string resultName = getName(op.getResult(), /*force=*/true);
  std::string sourceName = getName(op.getSource());

  auto getOffsetAsString = [&](OpFoldResult offset) -> std::string {
    if (auto attr = mlir::dyn_cast_or_null<Attribute>(offset)) {
      return std::to_string(
          mlir::cast<IntegerAttr>(attr).getValue().getSExtValue());
    }
    return getName(mlir::cast<Value>(offset));
  };

  if (llvm::all_of(op.getMixedStrides(), isConstantOne)) {
    bool contiguous = false;
    if (llvm::all_of(ArrayRef<OpFoldResult>(op.getMixedSizes()).drop_back(),
                     isConstantOne)) {
      contiguous = true;
    } else {
      contiguous = true;
      for (size_t i = 1; i < sourceType.getRank(); ++i) {
        auto size = getConstantIntValue(op.getMixedSizes()[i]);
        if (!size || *size != sourceType.getShape()[i]) {
          contiguous = false;
          break;
        }
      }
    }

    if (contiguous) {
      SmallVector<std::string> offsetStrings;
      for (auto o : op.getMixedOffsets()) {
        offsetStrings.push_back(getOffsetAsString(o));
      }
      std::string startStr =
          flattenIndexExpression(sourceType.getShape(), offsetStrings);

      std::string lengthStr;
      if (resultType.hasStaticShape()) {
        lengthStr = std::to_string(resultType.getNumElements());
      } else {
        SmallVector<std::string> sizeStrings;
        for (auto s : op.getMixedSizes()) {
          sizeStrings.push_back(getOffsetAsString(s));
        }
        lengthStr = sizeStrings[0];
        for (size_t i = 1; i < sizeStrings.size(); ++i) {
          lengthStr += " * " + sizeStrings[i];
        }
      }

      os << resultName << " := " << sourceName << "[" << startStr << " : "
         << startStr << " + " << lengthStr << "]\n";
      return success();
    }
  }

  // make an array to store the output
  //
  //   v := make([]int32, num_elements)
  //
  auto eltTypeStr = convertType(resultType.getElementType());
  if (resultType.hasStaticShape()) {
    os << resultName << " := make([]" << eltTypeStr.value() << ", "
       << resultType.getNumElements() << ")\n";
  } else {
    SmallVector<std::string> sizeStrings;
    for (auto s : op.getMixedSizes()) {
      sizeStrings.push_back(getOffsetAsString(s));
    }
    std::string lengthStr = sizeStrings[0];
    for (size_t i = 1; i < sizeStrings.size(); ++i) {
      lengthStr += " * " + sizeStrings[i];
    }
    os << resultName << " := make([]" << eltTypeStr.value() << ", " << lengthStr
       << ")\n";
  }

  SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = op.getMixedSizes();
  SmallVector<OpFoldResult> strides = op.getMixedStrides();

  InductionVarNameFn nameFn = [&](int i) {
    return getName(op.getResult()) + "_i" + std::to_string(i);
  };

  DynamicLoopOpenerFn opener = [&](raw_indented_ostream& os,
                                   const std::string& inductionVar,
                                   OpFoldResult size) {
    os << "for " << inductionVar << " := 0; " << inductionVar << " < "
       << getOffsetAsString(size) << "; " << inductionVar << " += 1 {\n";
  };

  emitFlattenedExtractSlice(resultType, sourceType, resultName, sourceName,
                            offsets, sizes, strides, getOffsetAsString, nameFn,
                            opener, os);

  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::CollapseShapeOp op) {
  // A rank-reduced type will have the same number of elements so collapsing
  // is a no-op on a flattened tensor.
  SliceVerificationResult res =
      isRankReducedType(op.getSrcType(), op.getResultType());
  if (res != SliceVerificationResult::Success) {
    return op.emitError()
           << "Only rank-reduced types are supported for CollapseShapeOp";
  }
  variableNames->mapValueNameToValue(op.getResult(), op.getSrc());
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::ExpandShapeOp op) {
  // A rank-reduced type will have the same number of elements so expanding is
  // a no-op on a flattened tensor.
  SliceVerificationResult res =
      isRankReducedType(op.getResultType(), op.getSrcType());
  if (res != SliceVerificationResult::Success) {
    return op.emitError()
           << "Only rank-reduced types are supported for ExpandShapeOp";
  }
  variableNames->mapValueNameToValue(op.getResult(), op.getSrc());
  return success();
}

// Emits a slice copy operation and returns the string containing the emitted
// variable name of the result
std::string LattigoEmitter::emitCopySlice(Value source, Value result,
                                          bool shouldDeclare) {
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
  std::string assign = shouldDeclare ? ":=" : "=";
  auto typeStr = convertType(result.getType());
  os << resultName << " " << assign << " append(make(" << typeStr.value()
     << ", 0, len(" << sourceName << ")), " << sourceName << "...)\n";
  return resultName;
}

LogicalResult LattigoEmitter::printOperation(tensor::InsertOp op) {
  // Check if the result was already declared (e.g., as an iter_arg in scf.for)
  bool shouldDeclare = !(variableNames->contains(op.getResult()) &&
                         variableNames->contains(op.getDest()) &&
                         getName(op.getResult()) == getName(op.getDest()));

  std::string resultName;
  if (!shouldDeclare) {
    resultName = getName(op.getResult(), /*force=*/true);
  } else {
    resultName = emitCopySlice(op.getDest(), op.getResult(), shouldDeclare);
  }
  // result[index] = value
  os << resultName << "[";
  os << flattenIndexExpression(op.getResult().getType(), op.getIndices(),
                               [&](Value value) { return getName(value); });
  os << "] = " << getName(op.getScalar()) << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::InsertSliceOp op) {
  RankedTensorType resultType = op.getResult().getType();
  if (resultType.getRank() != op.getSourceType().getRank()) {
    return op.emitError(
        "Lattigo emitter for InsertSliceOp only supports "
        "result rank equal to source rank");
  }

  // Check if the result was already declared (e.g., as an iter_arg in scf.for)
  bool shouldDeclare = !(variableNames->contains(op.getResult()) &&
                         variableNames->contains(op.getDest()) &&
                         getName(op.getResult()) == getName(op.getDest()));

  std::string destName;
  if (!shouldDeclare) {
    destName = getName(op.getResult(), /*force=*/true);
  } else {
    destName = emitCopySlice(op.getDest(), op.getResult(), shouldDeclare);
  }
  const std::string sourceName = getName(op.getSource());

  auto getOffsetAsString = [&](OpFoldResult offset) -> std::string {
    if (auto attr = mlir::dyn_cast_or_null<Attribute>(offset)) {
      return std::to_string(
          mlir::cast<IntegerAttr>(attr).getValue().getSExtValue());
    }
    return getName(mlir::cast<Value>(offset));
  };

  if (llvm::all_of(op.getMixedStrides(), isConstantOne)) {
    bool contiguous = false;
    if (llvm::all_of(ArrayRef<OpFoldResult>(op.getMixedSizes()).drop_back(),
                     isConstantOne)) {
      contiguous = true;
    } else {
      contiguous = true;
      for (size_t i = 1; i < op.getSourceType().getRank(); ++i) {
        auto size = getConstantIntValue(op.getMixedSizes()[i]);
        if (!size || *size != op.getSourceType().getShape()[i]) {
          contiguous = false;
          break;
        }
      }
    }

    if (contiguous) {
      SmallVector<std::string> offsetStrings;
      for (auto o : op.getMixedOffsets()) {
        offsetStrings.push_back(getOffsetAsString(o));
      }
      std::string startStr =
          flattenIndexExpression(resultType.getShape(), offsetStrings);

      std::string lengthStr;
      if (op.getSourceType().hasStaticShape()) {
        lengthStr = std::to_string(op.getSourceType().getNumElements());
      } else {
        SmallVector<std::string> sizeStrings;
        for (auto s : op.getMixedSizes()) {
          sizeStrings.push_back(getOffsetAsString(s));
        }
        lengthStr = sizeStrings[0];
        for (size_t i = 1; i < sizeStrings.size(); ++i) {
          lengthStr += " * " + sizeStrings[i];
        }
      }

      os << "copy(" << destName << "[" << startStr << " : " << startStr << " + "
         << lengthStr << "], " << sourceName << ")\n";
      return success();
    }
  }

  SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = op.getMixedSizes();
  SmallVector<OpFoldResult> strides = op.getMixedStrides();

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
    os << "for " << destIndexName
       << " := " << getOffsetAsString(offsets[nestLevel]) << "; "
       << destIndexName << " < " << getOffsetAsString(offsets[nestLevel])
       << " + " << getOffsetAsString(sizes[nestLevel]) << " * "
       << getOffsetAsString(strides[nestLevel]) << "; " << destIndexName
       << " += " << getOffsetAsString(strides[nestLevel]) << " {\n";
    os.indent();
  }

  // Now we're in the innermost loop nest, do the assignment
  os << destName << "["
     << flattenIndexExpression(resultType.getShape(), destIndexNames)
     << "] = " << sourceName << "["
     << flattenIndexExpression(op.getSourceType().getShape(), sourceIndexNames)
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
  auto typeStr = convertType(getElementTypeOrSelf(op.getResult().getType()));
  std::string resultName = getName(op.getResult());
  os << resultName << " := []" << typeStr.value() << "{";
  os << getCommaSeparatedNames(op.getOperands());
  os << "}\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::SplatOp op) {
  imports.insert(std::string(kSlicesImport));
  // don't use getName because the tmpvar is guaranteed to be used
  std::string resultName = getName(op.getResult());
  std::string tmpVar = resultName + "_0";
  auto scalarTypeString = convertType(op.getInput().getType());
  RankedTensorType resultType = op.getResult().getType();

  // golang requires a slice input to slices.Repeat
  // Note that all tensor values are flattened in this emitter.
  //
  //   numbers := []int{v}
  //   repeat := slices.Repeat(numbers, 50)
  //
  int tensorSize = resultType.getNumElements();
  os << tmpVar << " := []" << scalarTypeString.value() << "{"
     << getName(op.getInput()) << "}\n";
  os << resultName << " := slices.Repeat(" << tmpVar << ", " << tensorSize
     << ")\n";
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
  std::string resultName = getName(op.getOutput());
  os << resultName << " := " << getName(op.getInput()) << ".CopyNew()\n";
  os << getName(op.getEvaluator()) << ".DropLevel(" << resultName << ", "
     << op.getLevelToDrop() << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(RLWEDropLevelOp op) {
  if (getName(op.getOutput()) != getName(op.getInput())) {
    std::string resultName = getName(op.getOutput());
    os << resultName << " := " << getName(op.getInput()) << ".CopyNew()\n";
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
  std::string resultName = getName(op.getOutput());
  os << resultName << " := " << getName(op.getInput()) << ".CopyNew()\n";
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
  std::string resultName = getName(op.getResult());
  os << resultName;
  os << " := bgv.NewEvaluator(";
  os << getCommaSeparatedNames(operands);
  os << ", ";
  os << (op.getScaleInvariant() ? "true" : "false");
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVNewPlaintextOp op) {
  std::string resultName = getName(op.getResult());
  os << resultName << " := bgv.NewPlaintext(";
  os << getName(op.getParams()) << ", ";
  os << getName(op.getParams()) << ".MaxLevel()";
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVEncodeOp op) {
  // hack: access another op to get params
  auto newPlaintextOp =
      mlir::dyn_cast<BGVNewPlaintextOp>(op.getPlaintext().getDefiningOp());
  if (!newPlaintextOp) {
    return failure();
  }
  auto plaintextName = getName(op.getPlaintext());
  auto valueName = getName(op.getValue());
  auto maxSlotsName = getName(newPlaintextOp.getParams()) + ".MaxSlots()";
  auto numSlotsAttr = dyn_cast_or_null<IntegerAttr>(
      op->getParentOfType<ModuleOp>()->getAttr(kRequestedSlotCountAttrName));
  if (numSlotsAttr) {
    maxSlotsName = std::to_string(numSlotsAttr.getInt());
  }

  std::string packedName = valueName;
  // EncodeOp requires its argument to be a slice of int64 so we emit a loop
  // implementing type conversion if needed.
  if (getElementTypeOrSelf(op.getValue().getType()).getIntOrFloatBitWidth() !=
      64) {
    packedName = valueName + "_" + plaintextName + "_packed";
    os << packedName << " := make([]int64, ";
    os << maxSlotsName << ")\n";
    os << "for i := range " << packedName << " {\n";

    // packedName[i] = int64(value[i])
    auto valueNameAtI = valueName + "[i % len(" + valueName + ")]";
    auto packedNameAtI = packedName + "[i]";
    os.indent();
    if (getElementTypeOrSelf(op.getValue().getType()).getIntOrFloatBitWidth() ==
        1) {
      emitIf(
          valueNameAtI, [&]() { os << packedNameAtI << " = int64(1)\n"; },
          [&]() { os << packedNameAtI << " = int64(0)\n"; });
    } else {
      os << packedNameAtI << " = int64(" << valueNameAtI << ")\n";
    }
    os.unindent();
    os << "}\n";
  }

  // set the scale of plaintext
  auto scale = op.getScale();
  os << plaintextName << ".Scale = ";
  os << getName(newPlaintextOp.getParams()) << ".NewScale(";
  os << scale << ")\n";

  os << getName(op.getEncoder()) << ".Encode(";
  os << packedName << ", ";
  os << plaintextName << ")\n";
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
  auto convertedType = convertType(op.getDecoded().getType());
  os << convertedName << " := make(" << convertedType.value() << ", len("
     << getName(op.getValue()) << "))\n";
  os << "for i := range " << getName(op.getValue()) << " {\n";
  os.indent();
  auto elementConvertedType =
      convertType(getElementTypeOrSelf(op.getDecoded().getType()));
  os << convertedName << "[i] = " << elementConvertedType.value() << "("
     << valueInt64Name << "[i])\n";
  os.unindent();
  os << "}\n";
  std::string resultName = getName(op.getDecoded());
  os << resultName << " := " << convertedName << "\n";
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
  return printEvalInPlaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Add", true);
}

LogicalResult LattigoEmitter::printOperation(BGVSubOp op) {
  return printEvalInPlaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Sub", true);
}

LogicalResult LattigoEmitter::printOperation(BGVMulOp op) {
  return printEvalInPlaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Mul", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRelinearizeNewOp op) {
  return printEvalNewMethod(op.getOutput(), op.getEvaluator(), op.getInput(),
                            "RelinearizeNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRescaleNewOp op) {
  // there is no RescaleNew method in Lattigo, manually create new ciphertext
  std::string resultName = getName(op.getOutput());
  os << resultName << " := " << getName(op.getInput()) << ".CopyNew()\n";
  return printEvalInPlaceMethod(
      op.getEvaluator(), {op.getInput(), op.getOutput()}, "Rescale", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRotateColumnsNewOp op) {
  auto errName = getErrName();
  std::string resultName = getName(op.getOutput());
  os << resultName << ", " << errName << " := " << getName(op.getEvaluator())
     << ".RotateColumnsNew(";
  os << getName(op.getInput()) << ", ";

  // Handle both static_shift attribute and dynamic_shift SSA value
  if (op.getStaticShift().has_value()) {
    os << op.getStaticShift()->getInt();
  } else if (op.getDynamicShift()) {
    // RotateColumnsNew expects an int argument for the shift, and golang does
    // not allow implicit conversion from i32/i64
    os << "int(" << getName(op.getDynamicShift()) << ")";
  } else {
    return op.emitError(
        "RotateColumnsNewOp must have either static_shift attribute or "
        "dynamic_shift operand");
  }

  os << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVRotateRowsNewOp op) {
  return printEvalNewMethod(op.getOutput(), op.getEvaluator(), {op.getInput()},
                            "RotateRowsNew", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRelinearizeOp op) {
  return printEvalInPlaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "Relinearize", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRescaleOp op) {
  return printEvalInPlaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "Rescale", true);
}

LogicalResult LattigoEmitter::printOperation(BGVRotateColumnsOp op) {
  auto errName = getErrName();
  os << errName << " := " << getName(op.getEvaluator()) << ".RotateColumns(";
  os << getName(op.getInput()) << ", ";

  // Handle both static_shift attribute and dynamic_shift SSA value
  if (op.getStaticShift().has_value()) {
    os << op.getStaticShift()->getInt();
  } else if (op.getDynamicShift()) {
    // RotateColumnsNew expects an int argument for the shift, and golang does
    // not allow implicit conversion from i32/i64
    os << "int(" << getName(op.getDynamicShift()) << ")";
  } else {
    return op.emitError(
        "RotateColumnsOp must have either static_shift attribute or "
        "dynamic_shift operand");
  }

  os << ", ";
  os << getName(op.getInplace()) << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVRotateRowsOp op) {
  return printEvalInPlaceMethod(
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
  std::string resultName = getName(op.getResult());
  os << resultName << ", " << errName << " := bgv.NewParametersFromLiteral(";

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

LogicalResult LattigoEmitter::printOperation(
    CKKSGenEvaluationKeysBootstrappingOp op) {
  auto errName = getErrName();
  std::string resultName = getName(op.getResult());
  os << resultName << ", _, " << errName << " := " << getName(op.getParams())
     << ".GenEvaluationKeys(" << getName(op.getSk()) << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(
    CKKSNewBootstrappingEvaluatorOp op) {
  return printNewMethod(op.getResult(), op.getOperands(),
                        "bootstrapping.NewEvaluator", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSNewPlaintextOp op) {
  std::string resultName = getName(op.getResult());
  os << resultName << " := ckks.NewPlaintext(";
  os << getName(op.getParams()) << ", ";
  os << getName(op.getParams()) << ".MaxLevel()";
  os << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSEncodeOp op) {
  // hack: access another op to get params
  auto newPlaintextOp =
      mlir::dyn_cast<CKKSNewPlaintextOp>(op.getPlaintext().getDefiningOp());
  if (!newPlaintextOp) {
    return failure();
  }
  auto plaintextName = getName(op.getPlaintext());
  auto valueName = getName(op.getValue());
  auto maxSlotsName = getName(newPlaintextOp.getParams()) + ".MaxSlots()";
  auto numSlotsAttr = dyn_cast_or_null<IntegerAttr>(
      op->getParentOfType<ModuleOp>()->getAttr(kRequestedSlotCountAttrName));
  if (numSlotsAttr) {
    maxSlotsName = std::to_string(numSlotsAttr.getInt());
    imports.insert(std::string(kRingImport));
    int logNumSlots = (int)log2(numSlotsAttr.getInt());
    os << plaintextName
       << ".LogDimensions = ring.Dimensions{Rows: 0, Cols: " << logNumSlots
       << "}\n";
  }

  std::string packedName = valueName;
  // EncodeOp requires its argument to be a slice of int64 so we emit a loop
  // implementing type conversion if needed.
  if (getElementTypeOrSelf(op.getValue().getType()).getIntOrFloatBitWidth() !=
      64) {
    packedName = valueName + "_" + plaintextName + "_packed";
    os << packedName << " := make([]float64, ";
    os << maxSlotsName << ")\n";
    os << "for i := range " << packedName << " {\n";
    // packedName[i] = float64(value[i])
    auto valueNameAtI = valueName + "[i % len(" + valueName + ")]";
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
      os << packedNameAtI << " = float64(" << valueNameAtI << ")\n";
    }
    os.unindent();
    os << "}\n";
  }

  // set the scale of plaintext
  imports.insert(std::string(kMathImport));
  auto scale = op.getScale();
  os << plaintextName << ".Scale = ";
  os << getName(newPlaintextOp.getParams()) << ".NewScale(math.Pow(2, ";
  os << scale << "))\n";

  os << getName(op.getEncoder()) << ".Encode(";
  os << packedName << ", ";
  os << plaintextName << ")\n";
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
  auto convertedType = convertType(op.getDecoded().getType());
  os << convertedName << " := make(" << convertedType.value() << ", len("
     << getName(op.getValue()) << "))\n";
  os << "for i := range " << getName(op.getValue()) << " {\n";
  os.indent();
  auto elementConvertedType =
      convertType(getElementTypeOrSelf(op.getDecoded().getType()));
  os << convertedName << "[i] = " << elementConvertedType.value() << "("
     << valueFloat64Name << "[i])\n";
  os.unindent();
  os << "}\n";
  std::string resultName = getName(op.getDecoded());
  os << resultName << " := " << convertedName << "\n";
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
  return printEvalInPlaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Add", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSSubOp op) {
  return printEvalInPlaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Sub", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSMulOp op) {
  return printEvalInPlaceMethod(op.getEvaluator(),
                                {op.getLhs(), op.getRhs(), op.getInplace()},
                                "Mul", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRelinearizeNewOp op) {
  return printEvalNewMethod(op.getOutput(), op.getEvaluator(), op.getInput(),
                            "RelinearizeNew", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRescaleNewOp op) {
  // there is no RescaleNew method in Lattigo, manually create new ciphertext
  std::string resultName = getName(op.getOutput());
  os << resultName << " := " << getName(op.getInput()) << ".CopyNew()\n";
  return printEvalInPlaceMethod(
      op.getEvaluator(), {op.getInput(), op.getOutput()}, "Rescale", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRotateNewOp op) {
  auto errName = getErrName();
  std::string resultName = getName(op.getOutput());
  os << resultName << ", " << errName << " := " << getName(op.getEvaluator())
     << ".RotateNew(";
  os << getName(op.getInput()) << ", ";

  // Handle both static_shift attribute and dynamic_shift SSA value
  if (op.getStaticShift().has_value()) {
    os << op.getStaticShift()->getInt();
  } else if (op.getDynamicShift()) {
    // RotateColumnsNew expects an int argument for the shift, and golang does
    // not allow implicit conversion from i32/i64
    os << "int(" << getName(op.getDynamicShift()) << ")";
  } else {
    return op.emitError(
        "CKKSRotateNewOp must have either static_shift attribute or "
        "dynamic_shift operand");
  }

  os << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSRelinearizeOp op) {
  return printEvalInPlaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "Relinearize", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRescaleOp op) {
  return printEvalInPlaceMethod(
      op.getEvaluator(), {op.getInput(), op.getInplace()}, "Rescale", true);
}

LogicalResult LattigoEmitter::printOperation(CKKSRotateOp op) {
  auto errName = getErrName();
  os << errName << " := " << getName(op.getEvaluator()) << ".Rotate(";
  os << getName(op.getInput()) << ", ";

  // Handle both static_shift attribute and dynamic_shift SSA value
  if (op.getStaticShift().has_value()) {
    os << op.getStaticShift()->getInt();
  } else if (op.getDynamicShift()) {
    // RotateColumnsNew expects an int argument for the shift, and golang does
    // not allow implicit conversion from i32/i64
    os << "int(" << getName(op.getDynamicShift()) << ")";
  } else {
    return op.emitError(
        "CKKSRotateOp must have either static_shift attribute or "
        "dynamic_shift operand");
  }

  os << ", ";
  os << getName(op.getInplace()) << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSBootstrapOp op) {
  imports.insert(std::string(kBootstrappingImport));

  auto errName = getErrName();
  std::string resultName = getName(op.getResult());
  os << resultName << ", " << errName << " := " << getName(op.getEvaluator())
     << ".Bootstrap(";
  os << getName(op.getInput()) << ")\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSLinearTransformOp op) {
  imports.insert(std::string(kLintransImport));

  // Get the evaluator, input, and parameters from context
  auto evaluatorName = getName(op.getEvaluator());
  auto encoderName = getName(op.getEncoder());
  auto inputName = getName(op.getInput());
  auto outputName = getName(op.getOutput());
  auto diagonalsName = getName(op.getDiagonals());

  // Get the diagonals tensor type to determine dimensions
  auto diagonalsTensor = cast<RankedTensorType>(op.getDiagonals().getType());
  if (diagonalsTensor.getRank() != 2) {
    return op.emitOpError("Expected 2D tensor for diagonals");
  }

  int64_t slotsPerDiagonal = diagonalsTensor.getShape()[1];

  // Generate unique variable names
  std::string diagonalsMapName = outputName + "_diags";
  std::string diagonalIndices = outputName + "_diags_idx";
  std::string ltParamsName = outputName + "_params";
  std::string ltName = outputName + "_lt";
  std::string ltEvalName = outputName + "_lteval";
  std::string errName = getErrName();

  os << diagonalIndices
     << " := " << printDenseI32ArrayAttr(op.getDiagonalIndicesAttr()) << "\n";
  os << diagonalsMapName << " := make(lintrans.Diagonals[float64])\n";
  os << "for i, diagIndex := range " << diagonalIndices << " {\n";
  os.indent();
  os << diagonalsMapName << "[diagIndex] = " << diagonalsName << "[i*"
     << slotsPerDiagonal << ":(i+1)*" << slotsPerDiagonal << "]\n";
  os.unindent();
  os << "}\n";

  os << ltParamsName << " := lintrans.Parameters{\n";
  os.indent();
  os << "DiagonalsIndexList: " << diagonalsMapName
     << ".DiagonalsIndexList(),\n";
  os << "LevelQ: " << op.getLevelQ().getInt() << ",\n";
  os << "LevelP: " << evaluatorName << ".GetRLWEParameters().MaxLevelP(),\n";
  os << "Scale: rlwe.NewScale(" << evaluatorName << ".GetRLWEParameters().Q()["
     << op.getLevelQ().getInt() << "]),\n";
  os << "LogDimensions: " << inputName << ".LogDimensions,\n";
  os << "LogBabyStepGiantStepRatio: "
     << op.getLogBabyStepGiantStepRatio().getInt() << ",\n";
  os.unindent();
  os << "}\n";

  os << ltName << " := lintrans.NewTransformation(" << evaluatorName
     << ".GetRLWEParameters(), " << ltParamsName << ")\n";
  os << errName << " := lintrans.Encode[float64](" << encoderName << ", "
     << diagonalsMapName << ", " << ltName << ")\n";
  printErrPanic(errName);
  os << "\n";

  os << ltEvalName << " := lintrans.NewEvaluator(" << evaluatorName << ")\n";
  os << outputName << ", " << errName << " := " << ltEvalName
     << ".EvaluateNew(";
  os << inputName << ", " << ltName << ")\n";
  printErrPanic(errName);

  return success();
}

LogicalResult LattigoEmitter::printOperation(
    CKKSNewParametersFromLiteralOp op) {
  auto errName = getErrName();
  std::string resultName = getName(op.getResult());
  os << resultName << ", " << errName << " := ckks.NewParametersFromLiteral(";
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

LogicalResult LattigoEmitter::printOperation(
    CKKSNewBootstrappingParametersFromLiteralOp op) {
  imports.insert(std::string(kLattigoUtilsImport));
  auto btParams = op.getBtParamsLiteral();
  auto paramName = getName(op.getParams());
  auto errName = getErrName();
  auto numSlotsAttr = dyn_cast_or_null<IntegerAttr>(
      op->getParentOfType<ModuleOp>()->getAttr(kRequestedSlotCountAttrName));
  std::string resultName = getName(op.getResult());
  os << resultName << ", " << errName
     << " := bootstrapping.NewParametersFromLiteral(";
  os << paramName << ", ";
  os << "bootstrapping.ParametersLiteral{\n";
  os.indent();
  os << "LogN: utils.Pointy(" << btParams.getLogN() << "),\n";
  if (numSlotsAttr) {
    int numSlots = numSlotsAttr.getInt();
    int logNumSlots = (int)log2(numSlots);
    os << "LogSlots: utils.Pointy(" << logNumSlots << "),\n";
  }
  os.unindent();
  os << "})\n";
  printErrPanic(errName);
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSNewPolynomialEvaluatorOp op) {
  imports.insert(std::string(kCkksCircuitPolynomialImport));

  // Create polynomial evaluator using polynomial.NewEvaluator
  std::string resultName = getName(op.getResult());
  os << resultName << " := polynomial.NewEvaluator(";
  os << getName(op.getParams()) << ", ";
  os << getName(op.getEvaluator()) << ")\n";

  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSChebyshevOp op) {
  imports.insert(std::string(kMathBigImport));
  imports.insert(std::string(kLattigoBignumImport));

  std::string errName = getErrName();
  auto coeffsAttr = op.getCoefficients();

  std::string polyCoeffs = getName(op.getOutput()) + "_polyCoeffs";
  os << polyCoeffs << " := []*big.Float{\n";
  os.indent();
  for (auto coeff : coeffsAttr) {
    if (auto floatAttr = mlir::dyn_cast<FloatAttr>(coeff)) {
      double value = floatAttr.getValue().convertToDouble();
      // Format the float value with scientific notation
      std::ostringstream valueStream;
      valueStream << std::scientific << value;
      os << "new(big.Float).SetFloat64(" << valueStream.str() << "),\n";
    }
  }
  os.unindent();
  os << "}\n";
  std::string bignumPoly = getName(op.getOutput()) + "_bignumPoly";
  os << bignumPoly << " := bignum.NewPolynomial(bignum.Chebyshev, "
     << polyCoeffs
     << ", "
        "nil)\n";
  std::string resultName = getName(op.getOutput());
  os << resultName << ", " << errName << " := ";
  os << getName(op.getEvaluator()) << ".Evaluate(";
  os << getName(op.getCiphertext()) << ", ";
  os << bignumPoly << ", ";
  os << "rlwe.NewScale(" << op.getTargetScale().getInt() << "))\n";
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
  std::string resultName = getName(result);
  os << resultName;
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
LogicalResult LattigoEmitter::printEvalInPlaceMethod(
    ::mlir::Value evaluator, ::mlir::ValueRange operands, std::string_view op,
    bool err) {
  std::string errName = getErrName();
  if (err) {
    os << errName << " := ";
  }
  os << getName(evaluator) << "." << op << "("
     << getCommaSeparatedNames(operands) << ")\n";

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
      .Case<CKKSBootstrappingEvaluationKeysType>(
          [&](auto ty) { return std::string("*bootstrapping.EvaluationKeys"); })
      .Case<CKKSBootstrappingEvaluatorType>(
          [&](auto ty) { return std::string("*bootstrapping.Evaluator"); })
      .Case<CKKSParameterType>(
          [&](auto ty) { return std::string("ckks.Parameters"); })
      .Case<CKKSBootstrappingParameterType>(
          [&](auto ty) { return std::string("bootstrapping.Parameters"); })
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

std::string LattigoEmitter::declareVariable(Value value, Location loc) {
  std::string valueName = getName(value);
  if (!value.use_empty()) {
    os << "var " << getName(value) << " ";
    if (failed(emitType(value.getType()))) {
      emitError(loc, llvm::formatv("Failed to emit type for {}", value));
    }
    os << "\n";
  }
  return valueName;
}

LattigoEmitter::LattigoEmitter(raw_ostream& os,
                               SelectVariableNames* variableNames,
                               const std::string& packageName,
                               const std::vector<std::string>& extraImports,
                               std::function<bool(func::FuncOp)> funcFilter)
    : os(os),
      variableNames(variableNames),
      packageName(packageName),
      extraImports(extraImports),
      funcFilter(funcFilter) {}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
