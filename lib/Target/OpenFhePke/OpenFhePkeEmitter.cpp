#include "lib/Target/OpenFhePke/OpenFhePkeEmitter.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Target/OpenFhePke/OpenFhePkeTemplates.h"
#include "lib/Target/OpenFhePke/OpenFheUtils.h"
#include "lib/Target/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Polynomial/IR/PolynomialDialect.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

void registerToOpenFhePkeTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-openfhe-pke",
      "translate the openfhe dialect to C++ code against the OpenFHE pke API",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToOpenFhePke(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, func::FuncDialect,
                        openfhe::OpenfheDialect, lwe::LWEDialect,
                        ::mlir::polynomial::PolynomialDialect,
                        tensor::TensorDialect>();
      });
}

LogicalResult translateToOpenFhePke(Operation *op, llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  OpenFhePkeEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult OpenFhePkeEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp, arith::ExtSIOp, arith::IndexCastOp>(
              [&](auto op) { return printOperation(op); })
          // LWE ops
          .Case<lwe::RLWEDecodeOp, lwe::ReinterpretUnderlyingTypeOp>(
              [&](auto op) { return printOperation(op); })
          // OpenFHE ops
          .Case<AddOp, SubOp, MulNoRelinOp, MulOp, MulPlainOp, SquareOp,
                NegateOp, MulConstOp, RelinOp, ModReduceOp, LevelReduceOp,
                RotOp, AutomorphOp, KeySwitchOp, EncryptOp, DecryptOp,
                GenParamsOp, GenContextOp, GenMulKeyOp, GenRotKeyOp,
                MakePackedPlaintextOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(func::FuncOp funcOp) {
  if (funcOp.getNumResults() != 1) {
    return funcOp.emitOpError() << "Only functions with a single return type "
                                   "are supported, but this function has "
                                << funcOp.getNumResults();
    return failure();
  }

  Type result = funcOp.getResultTypes()[0];
  if (failed(emitType(result))) {
    return funcOp.emitOpError() << "Failed to emit type " << result;
  }

  os << " " << funcOp.getName() << "(";
  os.indent();

  // Check the types without printing to enable failure outside of
  // commaSeparatedValues; maybe consider making commaSeparatedValues combine
  // the results into a FailureOr, like commaSeparatedTypes in tfhe_rust
  // emitter.
  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType()))) {
      return funcOp.emitOpError() << "Failed to emit type " << arg.getType();
    }
  }

  os << commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
    return convertType(value.getType()).value() + " " +
           variableNames->getNameForValue(value);
  });
  os.unindent();
  os << ") {\n";
  os.indent();

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

LogicalResult OpenFhePkeEmitter::printOperation(func::ReturnOp op) {
  if (op.getNumOperands() != 1) {
    op.emitError() << "Only one return value supported";
    return failure();
  }
  os << "return " << variableNames->getNameForValue(op.getOperands()[0])
     << ";\n";
  return success();
}

void OpenFhePkeEmitter::emitAutoAssignPrefix(Value result) {
  // Use const auto& because most OpenFHE API methods would perform a copy
  // if using a plain `auto`.
  os << "const auto& " << variableNames->getNameForValue(result) << " = ";
}

LogicalResult OpenFhePkeEmitter::emitTypedAssignPrefix(Value result) {
  if (failed(emitType(result.getType()))) {
    return failure();
  }
  os << " " << variableNames->getNameForValue(result) << " = ";
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

LogicalResult OpenFhePkeEmitter::printOperation(SubOp op) {
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
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getCiphertext()}, "LevelReduce");
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
  auto result = convertType(op.getEvalKey().getType());
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

LogicalResult OpenFhePkeEmitter::printOperation(arith::ConstantOp op) {
  auto valueAttr = op.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    if (failed(emitTypedAssignPrefix(op.getResult()))) {
      return failure();
    }
    os << intAttr.getValue() << ";\n";
  } else if (auto denseElementsAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    if (denseElementsAttr.getType().getRank() != 1) {
      return op.emitError() << "Only 1D dense elements supported";
    }

    if (failed(emitTypedAssignPrefix(op.getResult()))) {
      return failure();
    }
    os << "{";

    auto cstIter = denseElementsAttr.value_begin<APInt>();
    auto cstIterEnd = denseElementsAttr.value_end<APInt>();
    SmallString<10> first;
    APInt firstVal = *cstIter;
    firstVal.toStringSigned(first);
    os << std::accumulate(std::next(cstIter), cstIterEnd, std::string(first),
                          [&](const std::string &a, const APInt &b) {
                            SmallString<10> str;
                            b.toStringSigned(str);
                            return a + ", " + std::string(str);
                          });
    os << "};\n";
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

LogicalResult OpenFhePkeEmitter::printOperation(arith::IndexCastOp op) {
  Type outputType = op.getOut().getType();
  if (failed(emitTypedAssignPrefix(op.getResult()))) {
    return failure();
  }
  os << "static_cast<";
  if (failed(emitType(outputType))) {
    return op.emitOpError() << "Unsupported index_cast op";
  }
  os << ">(" << variableNames->getNameForValue(op.getIn()) << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(
    lwe::ReinterpretUnderlyingTypeOp op) {
  emitAutoAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getInput()) << ";\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(
    openfhe::MakePackedPlaintextOp op) {
  std::string inputVarName = variableNames->getNameForValue(op.getValue());

  emitAutoAssignPrefix(op.getResult());
  FailureOr<Value> resultCC = getContextualCryptoContext(op.getOperation());
  if (failed(resultCC)) return resultCC;
  os << variableNames->getNameForValue(resultCC.value())
     << "->MakePackedPlaintext(" << inputVarName << ");\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(lwe::RLWEDecodeOp op) {
  // In OpenFHE a plaintext is already decoded by decrypt. The internal OpenFHE
  // implementation is simple enough (and dependent on currently-hard-coded
  // encoding choices) that we will eventually need to work at a lower level of
  // the API to support this operation properly.
  auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (tensorTy) {
    if (tensorTy.getRank() != 1) {
      return op.emitOpError() << "Only 1D tensors supported";
    }
    // OpenFHE plaintexts must be manually resized to the decoded output size
    // via plaintext->SetLength(<size>);
    auto size = tensorTy.getShape()[0];
    auto inputVarName = variableNames->getNameForValue(op.getInput());
    os << inputVarName << "->SetLength(" << size << ");\n";

    std::string tmpVar =
        variableNames->getNameForValue(op.getResult()) + "_cast";
    os << "const auto& " << tmpVar << " = ";
    os << inputVarName << "->GetPackedValue();\n";

    auto outputVarName = variableNames->getNameForValue(op.getResult());
    if (failed(emitType(tensorTy))) {
      return failure();
    }
    os << " " << outputVarName << "(std::begin(" << tmpVar << "), std::end("
       << tmpVar << "));\n";

    return success();
  }

  // By convention, a plaintext stores a scalar value in index 0
  auto result = emitTypedAssignPrefix(op.getResult());
  if (failed(result)) return result;
  os << variableNames->getNameForValue(op.getInput())
     << "->GetPackedValue()[0];\n";
  return success();
}

LogicalResult OpenFhePkeEmitter::printOperation(EncryptOp op) {
  return printEvalMethod(op.getResult(), op.getCryptoContext(),
                         {op.getPublicKey(), op.getPlaintext()}, "Encrypt");
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

  os << "CCParamsT " << paramsName << ";\n";
  os << paramsName << ".SetMultiplicativeDepth(" << mulDepth << ");\n";
  os << paramsName << ".SetPlaintextModulus(" << plainMod << ");\n";
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

LogicalResult OpenFhePkeEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

OpenFhePkeEmitter::OpenFhePkeEmitter(raw_ostream &os,
                                     SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
