#include "lib/Target/Lattigo/LattigoEmitter.h"

#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Lattigo/IR/LattigoDialect.h"
#include "lib/Dialect/Lattigo/IR/LattigoOps.h"
#include "lib/Dialect/Lattigo/IR/LattigoTypes.h"
#include "lib/Dialect/Mgmt/IR/MgmtDialect.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Target/Lattigo/LattigoTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/CommandLine.h"       // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/ManagedStatic.h"     // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lattigo {

LogicalResult translateToLattigo(Operation *op, llvm::raw_ostream &os,
                                 const std::string &packageName) {
  SelectVariableNames variableNames(op);
  LattigoEmitter emitter(os, &variableNames, packageName);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult LattigoEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp, func::CallOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // Tensor ops
          .Case<tensor::ExtractOp, tensor::FromElementsOp>(
              [&](auto op) { return printOperation(op); })
          // Lattigo ops
          .Case<
              // RLWE
              RLWENewEncryptorOp, RLWENewDecryptorOp, RLWENewKeyGeneratorOp,
              RLWEGenKeyPairOp, RLWEGenRelinearizationKeyOp, RLWEGenGaloisKeyOp,
              RLWENewEvaluationKeySetOp, RLWEEncryptOp, RLWEDecryptOp,
              RLWELevelReduceNewOp, RLWELevelReduceOp,
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
          .Default([&](Operation &) {
            return emitError(op.getLoc(), "unable to find printer for op");
          });

  if (failed(status)) {
    return emitError(op.getLoc(),
                     llvm::formatv("Failed to translate op {0}", op.getName()));
  }
  return success();
}

LogicalResult LattigoEmitter::printOperation(ModuleOp moduleOp) {
  os << "package " << packageName << "\n";

  if (moduleIsBGVOrBFV(moduleOp)) {
    os << kModulePreludeBGVTemplate;
  } else if (moduleIsCKKS(moduleOp)) {
    os << kModulePreludeCKKSTemplate;
  } else {
    return moduleOp.emitError("Unknown scheme");
  }

  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

bool LattigoEmitter::isDebugPort(StringRef debugPortName) {
  return debugPortName.rfind("__heir_debug") == 0;
}

StringRef LattigoEmitter::canonicalizeDebugPort(StringRef debugPortName) {
  if (isDebugPort(debugPortName)) {
    return "__heir_debug";
  }
  return debugPortName;
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
    if (auto *definingOp = ciphertext.getDefiningOp()) {
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

LogicalResult LattigoEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  std::string valueString;
  auto res =
      llvm::TypeSwitch<Attribute, LogicalResult>(valueAttr)
          .Case<IntegerAttr>([&](IntegerAttr intAttr) {
            valueString = std::to_string(intAttr.getInt());
            return success();
          })
          .Case<DenseElementsAttr>([&](DenseElementsAttr denseAttr) {
            if (succeeded(denseAttr.tryGetValues<APInt>())) {
              valueString = "[]int64{";
              for (auto value : denseAttr.getValues<APInt>()) {
                valueString += std::to_string(value.getSExtValue()) + ", ";
              }
            } else if (succeeded(denseAttr.tryGetValues<APFloat>())) {
              valueString = "[]float64{";
              for (auto value : denseAttr.getValues<APFloat>()) {
                valueString += std::to_string(value.convertToFloat()) + ", ";
              }
            } else {
              return failure();
            }
            // remote the trailing ", "
            if (valueString.size() > 1) {
              valueString.pop_back();
              valueString.pop_back();
            }
            valueString += "}";
            return success();
          })
          .Default([&](auto) { return failure(); });
  if (failed(res)) {
    return res;
  }
  os << getName(op.getResult()) << " := " << valueString << "\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::ExtractOp op) {
  // only support 1-dim tensor for now
  os << getName(op.getResult()) << " := " << getName(op.getTensor()) << "[";
  os << getName(op.getIndices()[0]) << "]\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(tensor::FromElementsOp op) {
  os << getName(op.getResult()) << " := []"
     << convertType(getElementTypeOrSelf(op.getResult().getType())) << "{";
  os << getCommaSeparatedNames(op.getOperands());
  os << "}\n";
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

LogicalResult LattigoEmitter::printOperation(RLWELevelReduceNewOp op) {
  // there is no LevelReduceNew method in Lattigo, manually create new
  // ciphertext
  os << getName(op.getOutput()) << " := " << getName(op.getInput())
     << ".CopyNew()\n";
  os << getName(op.getOutput()) << ".Resize(" << getName(op.getOutput())
     << ".Degree(), " << getName(op.getOutput()) << ".Level()-"
     << op.getLevelToDrop() << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(RLWELevelReduceOp op) {
  if (getName(op.getOutput()) != getName(op.getInput())) {
    os << getName(op.getInput()) << ".Copy(" << getName(op.getOutput())
       << ")\n";
  }
  os << getName(op.getOutput()) << ".Resize(" << getName(op.getOutput())
     << ".Degree(), " << getName(op.getOutput()) << ".Level()-"
     << op.getLevelToDrop() << ")\n";
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

  auto packedName =
      getName(op.getValue()) + "_" + getName(op.getPlaintext()) + "_packed";
  os << packedName << " := make([]int64, ";
  os << maxSlotsName << ")\n";
  os << "for i := range " << packedName << " {\n";
  os.indent();
  os << packedName << "[i] = int64(" << getName(op.getValue()) << "[i % len("
     << getName(op.getValue()) << ")])\n";
  os.unindent();
  os << "}\n";

  // set the scale of plaintext
  // Enable this part only when we have scale management
  // auto scale = op.getScale();
  // os << getName(op.getPlaintext()) << ".Scale = ";
  // os << getName(newPlaintextOp.getParams()) << ".NewScale(";
  // os << scale << ")\n";

  os << getName(op.getEncoder()) << ".Encode(";
  os << packedName << ", ";
  os << getName(op.getPlaintext()) << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(BGVDecodeOp op) {
  os << getName(op.getEncoder()) << ".Decode(";
  os << getName(op.getPlaintext()) << ", ";
  os << getName(op.getValue()) << ")\n";

  // type conversion from value to decoded
  auto convertedName = getName(op.getDecoded()) + "_converted";
  os << convertedName << " := make(" << convertType(op.getDecoded().getType())
     << ", len(" << getName(op.getValue()) << "))\n";
  os << "for i := range " << getName(op.getValue()) << " {\n";
  os.indent();
  os << convertedName
     << "[i] = " << convertType(getElementTypeOrSelf(op.getDecoded().getType()))
     << "(" << getName(op.getValue()) << "[i])\n";
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

  auto packedName =
      getName(op.getValue()) + "_" + getName(op.getPlaintext()) + "_packed";
  os << packedName << " := make([]float64, ";
  os << maxSlotsName << ")\n";
  os << "for i := range " << packedName << " {\n";
  os.indent();
  os << packedName << "[i] = float64(" << getName(op.getValue()) << "[i \% len("
     << getName(op.getValue()) << ")])\n";
  os.unindent();
  os << "}\n";

  // set the scale of plaintext
  // Enable this part only when we have scale management
  // auto scale = op.getScale();
  // os << getName(op.getPlaintext()) << ".Scale = ";
  // os << getName(newPlaintextOp.getParams()) << ".NewScale(math.Pow(2, ";
  // os << scale << "))\n";

  os << getName(op.getEncoder()) << ".Encode(";
  os << packedName << ", ";
  os << getName(op.getPlaintext()) << ")\n";
  return success();
}

LogicalResult LattigoEmitter::printOperation(CKKSDecodeOp op) {
  os << getName(op.getEncoder()) << ".Decode(";
  os << getName(op.getPlaintext()) << ", ";
  os << getName(op.getValue()) << ")\n";

  // type conversion from value to decoded
  auto convertedName = getName(op.getDecoded()) + "_converted";
  os << convertedName << " := make(" << convertType(op.getDecoded().getType())
     << ", len(" << getName(op.getValue()) << "))\n";
  os << "for i := range " << getName(op.getValue()) << " {\n";
  os.indent();
  os << convertedName
     << "[i] = " << convertType(getElementTypeOrSelf(op.getDecoded().getType()))
     << "(" << getName(op.getValue()) << "[i])\n";
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

LattigoEmitter::LattigoEmitter(raw_ostream &os,
                               SelectVariableNames *variableNames,
                               const std::string &packageName)
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
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToLattigo(op, output, translateOptions->packageName);
      },
      [](DialectRegistry &registry) {
        registry.insert<rns::RNSDialect, arith::ArithDialect, func::FuncDialect,
                        tensor::TensorDialect, lattigo::LattigoDialect,
                        mgmt::MgmtDialect>();
      });
}

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir
