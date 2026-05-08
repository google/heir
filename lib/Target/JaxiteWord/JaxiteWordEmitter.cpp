#include "lib/Target/JaxiteWord/JaxiteWordEmitter.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordDialect.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordOps.h"
#include "lib/Dialect/JaxiteWord/IR/JaxiteWordTypes.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Target/JaxiteWord/JaxiteWordTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Format.h"            // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
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
namespace jaxiteword {

void registerToJaxiteWordTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-jaxiteword",
      "translate the JaxiteWord dialect to python code for jaxiteword",
      [](Operation* op, llvm::raw_ostream& output) {
        return translateToJaxiteWord(op, output);
      },
      [](DialectRegistry& registry) {
        registry
            .insert<func::FuncDialect, jaxiteword::JaxiteWordDialect,
                    arith::ArithDialect, tensor::TensorDialect, scf::SCFDialect,
                    lwe::LWEDialect, memref::MemRefDialect>();
      });
}

LogicalResult translateToJaxiteWord(Operation* op, llvm::raw_ostream& os) {
  SelectVariableNames variableNames(op);
  JaxiteWordEmitter emitter(os, &variableNames);
  return emitter.translate(*op);
}

static int getMaxCurrentInModule(Operation* op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return 0;
  int maxCurrent = 0;
  module.walk([&](Operation* inner) {
    for (auto result : inner->getResults()) {
      if (auto ctType = dyn_cast<lwe::LWECiphertextType>(result.getType())) {
        maxCurrent =
            std::max(maxCurrent, ctType.getModulusChain().getCurrent());
      }
    }
    for (auto operand : inner->getOperands()) {
      if (auto ctType = dyn_cast<lwe::LWECiphertextType>(operand.getType())) {
        maxCurrent =
            std::max(maxCurrent, ctType.getModulusChain().getCurrent());
      }
    }
  });
  return maxCurrent;
}

static std::string getCrossLevelExpr(Value ct, StringRef ctxName, Operation* op,
                                     int extraOffset = 1) {
  auto ctType = cast<lwe::LWECiphertextType>(ct.getType());
  int current = ctType.getModulusChain().getCurrent();
  int maxCurrent = getMaxCurrentInModule(op);
  int rescalesFromFresh = maxCurrent - current;
  int totalOffset = rescalesFromFresh + extraOffset;
  if (totalOffset == 0) {
    return (ctxName + ".max_level").str();
  }
  if (totalOffset == 1) {
    return (ctxName + ".max_level - 1").str();
  }
  return (ctxName + ".max_level - " + Twine(totalOffset)).str();
}

static FailureOr<std::string> getConstantAsString(Value value) {
  if (auto constantOp =
          dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
    auto valueAttr = constantOp.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
      return std::to_string(intAttr.getInt());
    }
  }
  return failure();
}

LogicalResult JaxiteWordEmitter::translate(Operation& op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation&, LogicalResult>(op)
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          .Case<AddOp, SubOp, NegateOp, SquareOp, MulOp, MulNoRelinOp,
                ModReduceOp, RotOp, RelinOp, AddPlainOp, SubPlainOp, MulPlainOp,
                AddInPlaceOp, SubInPlaceOp, EncodeOp, DecodeOp, EncryptOp,
                DecryptOp, GenParamsOp, GenKeyPairOp, GenMulKeyOp, GenRotKeyOp,
                ProgramInitializationOp>(
              [&](auto op) { return printOperation(op); })
          .Case<tensor::ExtractOp, tensor::FromElementsOp, tensor::EmptyOp,
                tensor::InsertOp, tensor::ExtractSliceOp,
                tensor::InsertSliceOp>(
              [&](auto op) { return printOperation(op); })
          .Case<scf::ForOp, scf::IfOp, scf::YieldOp>(
              [&](auto op) { return printOperation(op); })
          .Case<memref::LoadOp, memref::StoreOp, memref::AllocOp>(
              [&](auto op) { return printOperation(op); })
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          .Case<arith::IndexCastOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
                arith::DivSIOp, arith::RemSIOp, arith::CmpIOp, arith::SelectOp,
                arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation&) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePrelude << "\n";
  for (Operation& op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(func::FuncOp funcOp) {
  os << "def " << funcOp.getName() << "(\n";
  os.indent();
  for (Value arg : funcOp.getArguments()) {
    auto argName = variableNames->getNameForValue(arg);
    os << argName << ": ";
    if (failed(emitType(arg.getType()))) {
      return funcOp.emitOpError()
             << "Failed to emit JaxiteWord type " << arg.getType();
    }
    os << ",\n";
    if (isa<lwe::LWECiphertextType>(arg.getType())) {
      CiphertextArg_ = argName;
    }
  }
  os.unindent();
  os << ")";

  if (funcOp.getNumResults() > 0) {
    os << " -> ";
    if (funcOp.getNumResults() == 1) {
      Type result = funcOp.getResultTypes()[0];
      if (failed(emitType(result))) {
        return funcOp.emitOpError()
               << "Failed to emit JaxiteWord type " << result;
      }
    } else {
      auto result = commaSeparatedTypes(
          funcOp.getResultTypes(), [&](Type type) -> FailureOr<std::string> {
            auto result = convertType(type);
            if (failed(result)) {
              return funcOp.emitOpError()
                     << "Failed to emit JaxiteWord type " << type;
            }
            return result;
          });
      os << "(" << result.value() << ")";
    }
  }

  os << ":\n";
  os.indent();

  for (Block& block : funcOp.getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }

  os.unindent();
  os << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(func::ReturnOp op) {
  std::function<std::string(Value)> resultValue = [&](Value value) {
    if (isa<BlockArgument>(value)) {
      return variableNames->getNameForValue(value);
    } else {
      return variableNames->getNameForValue(value);
    }
  };
  if (op.getNumOperands() == 0) {
    return success();
  }
  if (op.getNumOperands() == 1) {
    os << "return " << resultValue(op.getOperands()[0]) << "\n";
    return success();
  } else {
    os << "return (" << commaSeparatedValues(op.getOperands(), resultValue)
       << ")\n";
    return success();
  }
  return failure();
}

LogicalResult JaxiteWordEmitter::printOperation(AddOp op) {
  return printBinaryOpHelper(
      op.getResult(), op.getLhs(), op.getRhs(),
      [&](StringRef lhs, StringRef rhs, StringRef result) {
        os << lhs << "\n";
        os << llvm::formatv(kAddCoreTemplate.data(), result, rhs);
      });
}

LogicalResult JaxiteWordEmitter::printOperation(SubOp op) {
  return printBinaryOpHelper(
      op.getResult(), op.getLhs(), op.getRhs(),
      [&](StringRef lhs, StringRef rhs, StringRef result) {
        os << lhs << "\n";
        std::string rhsCiphertext = (rhs + ".ciphertext").str();
        os << llvm::formatv(kSubTemplate.data(), result, rhsCiphertext);
      });
}

LogicalResult JaxiteWordEmitter::printOperation(NegateOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getCiphertext()) << ".mul(-1)\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(SquareOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/1);

  emitAssignPrefix(op.getResult());
  os << ctx << ".he_mul[" << level << "].mul(" << ct << ", " << ct << ")\n";
  return success();
}

void JaxiteWordEmitter::emitAssignPrefix(Value result) {
  os << variableNames->getNameForValue(result) << " = ";
}

LogicalResult JaxiteWordEmitter::printBinaryOpHelper(
    Value result, Value lhs, Value rhs,
    llvm::function_ref<void(StringRef, StringRef, StringRef)> callback) {
  auto lhsName = variableNames->getNameForValue(lhs);
  auto rhsName = variableNames->getNameForValue(rhs);
  auto resultName = variableNames->getNameForValue(result);

  emitAssignPrefix(result);
  callback(lhsName, rhsName, resultName);
  return success();
}

LogicalResult JaxiteWordEmitter::printInPlaceBinaryOpHelper(
    Value lhs, Value rhs,
    llvm::function_ref<void(StringRef, StringRef)> callback) {
  auto lhsName = variableNames->getNameForValue(lhs);
  auto rhsName = variableNames->getNameForValue(rhs);

  callback(lhsName, rhsName);
  return success();
}

LogicalResult JaxiteWordEmitter::printMulOpHelper(
    Value result, Value lhs, Value rhs, Value ctx, Operation* op,
    llvm::function_ref<void(StringRef, StringRef, StringRef, StringRef,
                            StringRef)>
        callback) {
  auto lhsName = variableNames->getNameForValue(lhs);
  auto rhsName = variableNames->getNameForValue(rhs);
  auto ctxName = variableNames->getNameForValue(ctx);
  auto resultName = variableNames->getNameForValue(result);
  auto level = getCrossLevelExpr(lhs, ctxName, op, /*extraOffset=*/1);

  emitAssignPrefix(result);
  callback(lhsName, rhsName, ctxName, resultName, level);
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(EncodeOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getCryptoContext()) << ".encode("
     << variableNames->getNameForValue(op.getInput()) << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(EncryptOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto pk = variableNames->getNameForValue(op.getPublicKey());
  os << ctx << ".public_key = " << pk << "\n";

  emitAssignPrefix(op.getResult());
  os << ctx << ".encrypt(" << variableNames->getNameForValue(op.getPlaintext())
     << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(MulOp op) {
  return printMulOpHelper(op.getResult(), op.getLhs(), op.getRhs(),
                          op.getCryptoContext(), op.getOperation(),
                          [&](StringRef lhs, StringRef rhs, StringRef ctx,
                              StringRef result, StringRef level) {
                            os << ctx << ".he_mul[" << level << "].hemul("
                               << lhs << ", " << rhs << ")\n";
                          });
}

LogicalResult JaxiteWordEmitter::printOperation(MulNoRelinOp op) {
  return printMulOpHelper(op.getResult(), op.getLhs(), op.getRhs(),
                          op.getCryptoContext(), op.getOperation(),
                          [&](StringRef lhs, StringRef rhs, StringRef ctx,
                              StringRef result, StringRef level) {
                            os << ctx << ".he_mul[" << level
                               << "].hemul_no_relin(" << lhs << ", " << rhs
                               << ")\n";
                          });
}

LogicalResult JaxiteWordEmitter::printOperation(RelinOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/1);

  auto result = variableNames->getNameForValue(op.getOutput());
  os << llvm::formatv(kRelinTemplate.data(), result, ctx, level, ct);

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(ModReduceOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  emitAssignPrefix(op.getResult());
  os << ct << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(RotOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto rotIndex = op.getIndex();
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);

  emitAssignPrefix(op.getResult());
  os << ctx << ".he_rot[" << level << ", " << rotIndex << "].rotate(" << ct
     << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(DecryptOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto sk = variableNames->getNameForValue(op.getSecretKey());
  auto ct = variableNames->getNameForValue(op.getCiphertext());

  auto ctType = cast<lwe::LWECiphertextType>(op.getCiphertext().getType());
  int current = ctType.getModulusChain().getCurrent();
  int maxCurrent = getMaxCurrentInModule(op);
  int rescales = maxCurrent - current;

  os << llvm::formatv(kDecryptTemplate.data(), ctx, sk, rescales, ct);

  emitAssignPrefix(op.getResult());
  os << ctx << ".decrypt(_ct_for_dec)\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(DecodeOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getCryptoContext()) << ".decode("
     << variableNames->getNameForValue(op.getPlaintext())
     << ", is_ntt=False).real";

  if (auto tensorType = dyn_cast<RankedTensorType>(op.getResult().getType())) {
    os << ".reshape(";
    auto shape = tensorType.getShape();
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) os << ", ";
      os << shape[i];
    }
    os << ")";
  }
  os << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(MulPlainOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto pt = variableNames->getNameForValue(op.getPlaintext());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);

  os << ctx << ".ptct_mul[" << level << "].set_plaintext(" << pt << ")\n";
  emitAssignPrefix(op.getResult());
  os << ctx << ".ptct_mul[" << level << "].mul(" << ct << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(AddPlainOp op) {
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  os << lhs << ".ciphertext = " << lhs << ".ciphertext + " << rhs << "\n";
  os << llvm::formatv(kAddModReduceTemplate.data(), lhs);

  emitAssignPrefix(op.getResult());
  os << lhs << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(SubPlainOp op) {
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  os << llvm::formatv(kSubTemplate.data(), lhs, rhs);
  emitAssignPrefix(op.getResult());
  os << lhs << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(AddInPlaceOp op) {
  return printInPlaceBinaryOpHelper(
      op.getLhs(), op.getRhs(), [&](StringRef lhs, StringRef rhs) {
        os << llvm::formatv(kAddCoreTemplate.data(), lhs, rhs);
      });
}

LogicalResult JaxiteWordEmitter::printOperation(SubInPlaceOp op) {
  return printInPlaceBinaryOpHelper(
      op.getLhs(), op.getRhs(), [&](StringRef lhs, StringRef rhs) {
        std::string rhsCiphertext = (rhs + ".ciphertext").str();
        os << llvm::formatv(kSubTemplate.data(), lhs, rhsCiphertext);
      });
}

LogicalResult JaxiteWordEmitter::printOperation(GenKeyPairOp op) {
  auto pk = variableNames->getNameForValue(op.getPublicKey());
  auto sk = variableNames->getNameForValue(op.getSecretKey());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());

  os << "key_pair = key_gen.gen_pke_pair(" << ctx << ".q_towers, " << ctx
     << ".p_towers, " << "degree=" << ctx << ".degree" << ")\n";
  os << sk << " = key_pair[\"secret_key\"]\n";
  os << pk << " = key_pair[\"public_key\"]\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(GenMulKeyOp op) {
  auto ek = variableNames->getNameForValue(op.getResult());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto sk = variableNames->getNameForValue(op.getSecretKey());

  os << ek << " = key_gen.gen_evaluation_key(" << sk << ", " << "q=" << ctx
     << ".q_towers, " << "P=" << ctx << ".p_towers, " << "dnum=" << ctx
     << ".parameters.get('dnum', 3)" << ")\n";

  heMulVarName_ = "he_mul";
  os << llvm::formatv(kGenMulKeyTemplate.data(), ctx, heMulVarName_, ek);

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(GenRotKeyOp op) {
  auto rk = variableNames->getNameForValue(op.getResult());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto sk = variableNames->getNameForValue(op.getSecretKey());

  rotKeysDictVarName_ = rk + "_dict";
  std::string indicesStr;
  llvm::raw_string_ostream indicesOs(indicesStr);
  llvm::interleaveComma(op.getIndices(), indicesOs);

  os << llvm::formatv(kGenRotKeyTemplate.data(), rotKeysDictVarName_,
                      indicesStr, sk, ctx, heRotVarName_, rk);

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(GenParamsOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  cryptoContextVarName_ = ctx;

  os << "params = {\n";
  os.indent();
  auto qTowers = op.getQTowers();
  auto pTowers = op.getPTowers();
  int32_t compDeg = op.getCompositeDegree();

  os << "\"degree\": " << op.getDegree() << ",\n";
  os << "\"num_slots\": " << op.getNumSlots() << ",\n";
  os << "\"batch\": " << op.getBatch() << ",\n";
  os << "\"r\": " << op.getR() << ",\n";
  os << "\"c\": " << op.getC() << ",\n";
  os << "\"dnum\": " << op.getDnum() << ",\n";
  os << "\"numEvalMult\": " << op.getNumEvalMult() << ",\n";

  os << "\"q_towers\": [";
  llvm::interleaveComma(qTowers, os);
  os << "],\n";

  os << "\"p_towers\": [";
  llvm::interleaveComma(pTowers, os);
  os << "],\n";

  os << "\"composite_degree\": " << compDeg << ",\n";
  os << "\"p\": 30,\n";
  os << "\"max_bits_in_word\": 61,\n";
  os << "\"max_bits_value\": " << ((1ULL << 63) - (1ULL << 9) - 1) << ",\n";
  os << "\"noise_scale_degree\": 1,\n";
  os << "\"CKKS_M_FACTOR\": 1\n";
  os.unindent();
  os << "}\n";

  os << ctx << " = ckks.CKKSContext(params)\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(ProgramInitializationOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto sk = variableNames->getNameForValue(op.getSecretKey());

  os << ctx << ".secret_key = " << sk << "\n";
  os << ctx << ".program_initialization(";
  os << "total_hemul_levels=" << op.getTotalHemulLevels() << ", ";

  os << "total_rotation_indices=[";
  auto rotIndices = op.getTotalRotationIndices();
  llvm::interleaveComma(rotIndices, os);
  os << "], ";

  os << "dnum=" << op.getDnum() << ", ";
  os << "r=" << op.getR() << ", ";
  os << "c=" << op.getC() << ", ";
  os << "batch=" << op.getBatch();
  os << ")\n";

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(tensor::ExtractOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getTensor());

  os << "[";
  for (size_t i = 0; i < op.getIndices().size(); ++i) {
    if (i > 0) os << ", ";
    Value idx = op.getIndices()[i];
    auto constStr = getConstantAsString(idx);
    if (succeeded(constStr)) {
      os << constStr.value();
    } else {
      os << variableNames->getNameForValue(idx);
    }
  }
  os << "]\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(tensor::FromElementsOp op) {
  if (op.getNumOperands() == 0) {
    return success();
  }
  if (isa<jaxiteword::AddOp>(op->getOperands()[0].getDefiningOp())) {
    return success();
  }
  emitAssignPrefix(op.getResult());
  os << "[" << commaSeparatedValues(op.getOperands(), [&](Value value) {
    return variableNames->getNameForValue(value);
  }) << "]\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(memref::LoadOp op) {
  emitAssignPrefix(op.getResult());
  os << variableNames->getNameForValue(op.getMemref());
  if (isa<BlockArgument>(op.getMemref())) {
    os << "["
       << flattenedIndex(
              op.getMemRefType(), op.getIndices(),
              [&](Value value) {
                return dyn_cast<IntegerAttr>(
                           dyn_cast<arith::ConstantOp>(value.getDefiningOp())
                               .getValue())
                    .getValue()
                    .getSExtValue();
              })
       << "]";
  } else {
    os << bracketEnclosedValues(op.getIndices(), [&](Value value) {
      SmallString<16> idx_str;
      dyn_cast<IntegerAttr>(
          dyn_cast<arith::ConstantOp>(value.getDefiningOp()).getValue())
          .getValue()
          .toStringUnsigned(idx_str);
      return std::string(idx_str);
    });
  }
  os << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(memref::AllocOp op) {
  emitAssignPrefix(op.getResult());
  os << "jnp.full(("
     << std::accumulate(std::next(op.getMemref().getType().getShape().begin()),
                        op.getMemref().getType().getShape().end(),
                        std::to_string(op.getMemref().getType().getShape()[0]),
                        [&](const std::string& a, int64_t b) {
                          return a + "*" + std::to_string(b);
                        })
     << "), None)";
  os << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(memref::StoreOp op) {
  os << variableNames->getNameForValue(op.getMemref());
  os << "["
     << flattenedIndex(
            op.getMemRefType(), op.getIndices(),
            [&](Value value) {
              return dyn_cast<IntegerAttr>(
                         dyn_cast<arith::ConstantOp>(value.getDefiningOp())
                             .getValue())
                  .getValue()
                  .getSExtValue();
            })
     << "]";
  os << " = " << variableNames->getNameForValue(op.getValueToStore());
  os << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(tensor::EmptyOp op) {
  RankedTensorType resultType = op.getResult().getType();
  auto shape = resultType.getShape();
  Type elemType = resultType.getElementType();

  if (isa<lwe::LWECiphertextType>(elemType) ||
      isa<lwe::LWEPlaintextType>(elemType)) {
    emitAssignPrefix(op.getResult());
    os << "[None] * ";
    int64_t totalSize = 1;
    for (auto dim : shape) totalSize *= dim;
    os << totalSize << "\n";
    return success();
  }

  os << " = np.zeros((";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) os << ", ";
    os << shape[i];
  }
  os << ",)";

  if (elemType.isF32()) {
    os << ", dtype=np.float32";
  } else if (elemType.isF64()) {
    os << ", dtype=np.float64";
  } else if (elemType.isInteger(32)) {
    os << ", dtype=np.int32";
  } else if (elemType.isInteger(64)) {
    os << ", dtype=np.int64";
  }
  os << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(tensor::InsertOp op) {
  std::string destName = variableNames->getNameForValue(op.getDest());
  std::string resultName = variableNames->getNameForValue(op.getResult());
  std::string scalarName = variableNames->getNameForValue(op.getScalar());

  os << destName << "[";
  for (size_t i = 0; i < op.getIndices().size(); ++i) {
    if (i > 0) os << ", ";
    Value idx = op.getIndices()[i];
    auto constStr = getConstantAsString(idx);
    if (succeeded(constStr)) {
      os << constStr.value();
    } else {
      os << variableNames->getNameForValue(idx);
    }
  }
  os << "] = " << scalarName << "\n";

  if (resultName != destName) {
    os << resultName << " = " << destName << "\n";
  }
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(tensor::ExtractSliceOp op) {
  RankedTensorType resultType = op.getResult().getType();
  std::string resultName = variableNames->getNameForValue(op.getResult());
  std::string sourceName = variableNames->getNameForValue(op.getSource());

  SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  ArrayRef<int64_t> sizes = op.getStaticSizes();
  ArrayRef<int64_t> strides = op.getStaticStrides();

  os << resultName << " = " << sourceName << "[";
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (i > 0) os << ", ";

    std::string offsetStr;
    if (auto intAttr = offsets[i].dyn_cast<Attribute>()) {
      offsetStr =
          std::to_string(cast<IntegerAttr>(intAttr).getValue().getSExtValue());
    } else {
      offsetStr = variableNames->getNameForValue(cast<Value>(offsets[i]));
    }

    int64_t size = sizes[i];
    int64_t stride = strides[i];

    os << offsetStr << ":" << offsetStr << " + " << size;
    if (stride != 1) {
      os << ":" << stride;
    }
  }
  os << "]";

  if (resultType.getRank() < (int64_t)offsets.size()) {
    os << ".reshape(";
    auto shape = resultType.getShape();
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) os << ", ";
      os << shape[i];
    }
    os << ")";
  }
  os << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(tensor::InsertSliceOp op) {
  std::string destName = variableNames->getNameForValue(op.getDest());
  std::string resultName = variableNames->getNameForValue(op.getResult());
  std::string sourceName = variableNames->getNameForValue(op.getSource());

  os << resultName << " = " << destName << ".copy()\n";

  SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  ArrayRef<int64_t> sizes = op.getStaticSizes();
  ArrayRef<int64_t> strides = op.getStaticStrides();

  os << resultName << "[";
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (i > 0) os << ", ";

    std::string offsetStr;
    if (auto intAttr = offsets[i].dyn_cast<Attribute>()) {
      offsetStr =
          std::to_string(cast<IntegerAttr>(intAttr).getValue().getSExtValue());
    } else {
      offsetStr = variableNames->getNameForValue(cast<Value>(offsets[i]));
    }

    int64_t size = sizes[i];
    int64_t stride = strides[i];

    os << offsetStr << ":" << offsetStr << " + " << size;
    if (stride != 1) {
      os << ":" << stride;
    }
  }
  os << "] = " << sourceName << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(scf::ForOp op) {
  for (auto i = 0; i < op.getNumRegionIterArgs(); ++i) {
    Value result = op.getResults()[i];
    Value operand = op.getInitArgs()[i];
    Value iterArg = op.getRegionIterArgs()[i];
    Value yieldedValue = op.getYieldedValues()[i];

    std::string resultName = variableNames->getNameForValue(result);
    std::string initName = variableNames->getNameForValue(operand);

    variableNames->mapValueNameToValue(iterArg, result);
    variableNames->mapValueNameToValue(yieldedValue, result);

    if (resultName == initName) {
      continue;
    }

    emitAssignPrefix(result);
    os << initName;
    if (isa<ShapedType>(result.getType())) {
      os << ".copy()";
    }
    os << "\n";
  }

  auto getLbOrUb = [&](Value value) -> std::string {
    auto constStr = getConstantAsString(value);
    if (succeeded(constStr)) {
      return constStr.value();
    }
    return variableNames->getNameForValue(value);
  };

  std::string lb = getLbOrUb(op.getLowerBound());
  std::string ub = getLbOrUb(op.getUpperBound());
  std::string step = getLbOrUb(op.getStep());
  std::string inductionVar =
      variableNames->getNameForValue(op.getInductionVar());

  os << "for " << inductionVar << " in range(" << lb << ", " << ub;
  if (step != "1") {
    os << ", " << step;
  }
  os << "):\n";
  os.indent();

  for (Operation& bodyOp : *op.getBody()) {
    if (failed(translate(bodyOp))) {
      return bodyOp.emitOpError() << "Failed to translate for loop body";
    }
  }

  os.unindent();
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(scf::IfOp op) {
  for (auto i = 0; i < op.getNumResults(); ++i) {
    Value result = op.getResults()[i];
    Value thenYielded = op.thenYield().getResults()[i];
    Value elseYielded = op.elseYield().getResults()[i];

    variableNames->mapValueNameToValue(thenYielded, result);
    variableNames->mapValueNameToValue(elseYielded, result);
  }

  os << "if " << variableNames->getNameForValue(op.getCondition()) << ":\n";
  os.indent();
  for (Operation& thenOp : *op.thenBlock()) {
    if (failed(translate(thenOp))) {
      return thenOp.emitOpError() << "Failed to translate if then block";
    }
  }
  os.unindent();

  os << "else:\n";
  os.indent();
  for (Operation& elseOp : *op.elseBlock()) {
    if (failed(translate(elseOp))) {
      return elseOp.emitOpError() << "Failed to translate if else block";
    }
  }
  os.unindent();

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(scf::YieldOp op) {
  if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
    for (auto i = 0; i < op.getNumOperands(); ++i) {
      Value operand = op.getOperands()[i];
      Value result = ifOp.getResults()[i];
      if (!isa<ShapedType>(result.getType())) {
        emitAssignPrefix(result);
        os << variableNames->getNameForValue(operand) << "\n";
      }
    }
  }
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::ConstantOp op) {
  std::string varName = variableNames->getNameForValue(op.getResult());
  Attribute valueAttr = op.getValue();

  if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
    os << varName << " = " << intAttr.getInt() << "\n";
  } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    os << varName << " = " << floatAttr.getValueAsDouble() << "\n";
  } else if (auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr)) {
    auto shapedType = denseAttr.getType();
    auto shape = shapedType.getShape();
    Type elemType = shapedType.getElementType();

    if (denseAttr.isSplat()) {
      std::string dtype = "np.float32";
      if (elemType.isF64())
        dtype = "np.float64";
      else if (elemType.isInteger(16))
        dtype = "np.int16";
      else if (elemType.isInteger(32))
        dtype = "np.int32";
      else if (elemType.isInteger(64))
        dtype = "np.int64";

      os << varName << " = np.full((";
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) os << ", ";
        os << shape[i];
      }
      os << ",), ";

      if (elemType.isF32() || elemType.isF64()) {
        os << denseAttr.getSplatValue<APFloat>().convertToDouble();
      } else if (elemType.isInteger(16) || elemType.isInteger(32) ||
                 elemType.isInteger(64)) {
        os << denseAttr.getSplatValue<APInt>().getSExtValue();
      } else {
        os << "0";
      }
      os << ", dtype=" << dtype << ")\n";
    } else {
      os << varName << " = np.array([";

      bool first = true;
      if (elemType.isF32() || elemType.isF64()) {
        for (auto val : denseAttr.getValues<APFloat>()) {
          if (!first) os << ", ";
          first = false;
          os << val.convertToDouble();
        }
      } else if (elemType.isInteger(16) || elemType.isInteger(32) ||
                 elemType.isInteger(64)) {
        for (auto val : denseAttr.getValues<APInt>()) {
          if (!first) os << ", ";
          first = false;
          os << val.getSExtValue();
        }
      }

      os << "]";
      std::string dtype = "np.float32";
      if (elemType.isF64())
        dtype = "np.float64";
      else if (elemType.isInteger(16))
        dtype = "np.int16";
      else if (elemType.isInteger(32))
        dtype = "np.int32";
      else if (elemType.isInteger(64))
        dtype = "np.int64";
      os << ", dtype=" << dtype << ")";

      if (shape.size() > 1) {
        os << ".reshape(";
        for (size_t i = 0; i < shape.size(); ++i) {
          if (i > 0) os << ", ";
          os << shape[i];
        }
        os << ")";
      }
      os << "\n";
    }
  } else {
    os << varName << " = 0  # placeholder\n";
  }

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::IndexCastOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = int("
     << variableNames->getNameForValue(op.getIn()) << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::AddIOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = "
     << variableNames->getNameForValue(op.getLhs()) << " + "
     << variableNames->getNameForValue(op.getRhs()) << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::SubIOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = "
     << variableNames->getNameForValue(op.getLhs()) << " - "
     << variableNames->getNameForValue(op.getRhs()) << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::MulIOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = "
     << variableNames->getNameForValue(op.getLhs()) << " * "
     << variableNames->getNameForValue(op.getRhs()) << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::DivSIOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = "
     << variableNames->getNameForValue(op.getLhs()) << " // "
     << variableNames->getNameForValue(op.getRhs()) << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::RemSIOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = "
     << variableNames->getNameForValue(op.getLhs()) << " % "
     << variableNames->getNameForValue(op.getRhs()) << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::CmpIOp op) {
  std::string cmpOp;
  switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      cmpOp = "==";
      break;
    case arith::CmpIPredicate::ne:
      cmpOp = "!=";
      break;
    case arith::CmpIPredicate::slt:
      cmpOp = "<";
      break;
    case arith::CmpIPredicate::sle:
      cmpOp = "<=";
      break;
    case arith::CmpIPredicate::sgt:
      cmpOp = ">";
      break;
    case arith::CmpIPredicate::sge:
      cmpOp = ">=";
      break;
    case arith::CmpIPredicate::ult:
      cmpOp = "<";
      break;
    case arith::CmpIPredicate::ule:
      cmpOp = "<=";
      break;
    case arith::CmpIPredicate::ugt:
      cmpOp = ">";
      break;
    case arith::CmpIPredicate::uge:
      cmpOp = ">=";
      break;
  }
  os << variableNames->getNameForValue(op.getResult()) << " = "
     << variableNames->getNameForValue(op.getLhs()) << " " << cmpOp << " "
     << variableNames->getNameForValue(op.getRhs()) << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::SelectOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = "
     << variableNames->getNameForValue(op.getTrueValue()) << " if "
     << variableNames->getNameForValue(op.getCondition()) << " else "
     << variableNames->getNameForValue(op.getFalseValue()) << "\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::ExtSIOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = int("
     << variableNames->getNameForValue(op.getIn()) << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::ExtUIOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = int("
     << variableNames->getNameForValue(op.getIn()) << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(arith::TruncIOp op) {
  os << variableNames->getNameForValue(op.getResult()) << " = int("
     << variableNames->getNameForValue(op.getIn()) << ")\n";
  return success();
}

FailureOr<std::string> JaxiteWordEmitter::convertType(Type type) {
  if (type.isF64() || type.isF32()) {
    return std::string("float");
  }
  if (type.isInteger(1)) {
    return std::string("bool");
  }
  if (type.isInteger(16) || type.isInteger(32) || type.isInteger(64) ||
      type.isIndex()) {
    return std::string("int");
  }

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return std::string("np.ndarray");
  }

  return llvm::TypeSwitch<Type, FailureOr<std::string>>(type)
      .Case<lwe::LWECiphertextType>(
          [&](auto) { return std::string("Ciphertext"); })
      .Case<PublicKeyType>([&](auto) { return std::string("np.ndarray"); })
      .Case<PrivateKeyType>([&](auto) { return std::string("np.ndarray"); })
      .Case<EvalKeyType>([&](auto) { return std::string("dict"); })
      .Case<CryptoContextType>(
          [&](auto) { return std::string("ckks.CKKSContext"); })
      .Case<lwe::LWEPlaintextType>(
          [&](auto) { return std::string("Ciphertext"); })
      .Default([&](Type) { return failure(); });
}

LogicalResult JaxiteWordEmitter::emitType(Type type) {
  auto result = convertType(type);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

JaxiteWordEmitter::JaxiteWordEmitter(raw_ostream& os,
                                     SelectVariableNames* variableNames)
    : os(os), variableNames(variableNames) {}

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir
