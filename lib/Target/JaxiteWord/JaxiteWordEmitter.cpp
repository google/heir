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
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project
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
          .Case<func::FuncOp, func::CallOp, func::ReturnOp>(
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
  os << kEnsurePolyHelper << "\n";
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

LogicalResult JaxiteWordEmitter::printOperation(func::CallOp op) {
  if (op.getNumResults() == 1) {
    emitAssignPrefix(op.getResult(0));
  } else if (op.getNumResults() > 1) {
    os << "(";
    for (auto [idx, result] : llvm::enumerate(op.getResults())) {
      if (idx > 0) os << ", ";
      os << variableNames->getNameForValue(result);
    }
    os << ") = ";
  }

  os << op.getCallee().str() << "(";
  os << commaSeparatedValues(op.getOperands(), [&](Value value) {
    return variableNames->getNameForValue(value);
  });
  os << ")\n";
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
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  auto result = variableNames->getNameForValue(op.getResult());

  emitModularAdd(result, ctx, lhs, rhs);
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(SubOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  auto result = variableNames->getNameForValue(op.getResult());
  std::string rhsWork = (result + "_rhs");

  emitNormalizeCiphertext(result, ctx, lhs);
  emitNormalizeCiphertext(rhsWork, ctx, rhs);
  os << "_moduli = jnp.array(" << result << ".moduli, dtype=jnp.uint32)\n";
  os << result << ".polynomial = jnp.where(" << result << ".polynomial < "
     << rhsWork << ".polynomial, " << result << ".polynomial + _moduli - "
     << rhsWork << ".polynomial, " << result << ".polynomial - " << rhsWork
     << ".polynomial)\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(NegateOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto result = variableNames->getNameForValue(op.getResult());

  emitNormalizeCiphertext(result, ctx, ct);
  os << "_moduli = jnp.array(" << result << ".moduli, dtype=jnp.uint32)\n";
  os << result << ".polynomial = jnp.where(" << result << ".polynomial == 0, "
     << result << ".polynomial, _moduli - " << result << ".polynomial)\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(SquareOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto result = variableNames->getNameForValue(op.getResult());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/1);
  std::string inputLevel = ("(" + level + ") + 1");
  std::string ctWork = (result + "_arg");

  emitNormalizeCiphertext(ctWork, ctx, ct, inputLevel);
  os << result << " = " << ctx << ".he_mul[" << level << "].mul(" << ctWork
     << ", " << ctWork << ")\n";
  return success();
}

void JaxiteWordEmitter::emitAssignPrefix(Value result) {
  os << variableNames->getNameForValue(result) << " = ";
}

void JaxiteWordEmitter::emitAssignCiphertext(StringRef targetName,
                                             StringRef sourceName) {
  os << "_assign_poly(" << targetName << ", " << sourceName << ")\n";
}

void JaxiteWordEmitter::emitNormalizeCiphertext(StringRef resultName,
                                                StringRef ctxName,
                                                StringRef sourceName,
                                                StringRef levelExpr) {
  os << resultName << " = _ensure_poly(" << ctxName << ", " << sourceName;
  if (!levelExpr.empty()) os << ", " << levelExpr;
  os << ")\n";
}

void JaxiteWordEmitter::emitModularAdd(StringRef resultName, StringRef ctxName,
                                       StringRef lhsName, StringRef rhsName) {
  std::string lhsData = (resultName + "_lhs").str();
  std::string rhsData = (resultName + "_rhs").str();
  std::string numModuli = (resultName + "_num_moduli").str();
  std::string moduliSrc = (resultName + "_moduli_src").str();
  std::string moduli = (resultName + "_moduli").str();
  std::string sum = (resultName + "_sum").str();

  os << lhsData << " = " << lhsName << ".polynomial if hasattr(" << lhsName
     << ", \"polynomial\") else " << lhsName << "\n";
  os << rhsData << " = " << rhsName << ".polynomial if hasattr(" << rhsName
     << ", \"polynomial\") else " << rhsName << "\n";
  os << lhsData << " = " << lhsData << ".reshape(" << lhsData << ".shape[0], "
     << lhsData << ".shape[1], " << ctxName << "._param_cache.r, " << ctxName
     << "._param_cache.c, " << lhsData << ".shape[-1])\n";
  os << rhsData << " = " << rhsData << ".reshape(" << rhsData << ".shape[0], "
     << rhsData << ".shape[1], " << ctxName << "._param_cache.r, " << ctxName
     << "._param_cache.c, " << rhsData << ".shape[-1])\n";
  os << "if " << lhsData << ".shape != " << rhsData << ".shape:\n";
  os.indent();
  os << "raise ValueError(\"ciphertext add shape mismatch\")\n";
  os.unindent();
  os << numModuli << " = " << lhsData << ".shape[-1]\n";
  os << "if hasattr(" << lhsName << ", \"moduli\") and hasattr(" << rhsName
     << ", \"moduli\"):\n";
  os.indent();
  os << "if list(" << lhsName << ".moduli)[:" << numModuli << "] != list("
     << rhsName << ".moduli)[:" << numModuli << "]:\n";
  os.indent();
  os << "raise ValueError(\"ciphertext add modulus mismatch\")\n";
  os.unindent();
  os.unindent();
  os << moduliSrc << " = getattr(" << lhsName << ", \"moduli\", getattr("
     << rhsName << ", \"moduli\", " << ctxName << ".q_towers))\n";
  os << "if isinstance(" << moduliSrc << ", (int, np.integer)):\n";
  os.indent();
  os << moduliSrc << " = [" << moduliSrc << "]\n";
  os.unindent();
  os << moduli << " = jnp.array(list(" << moduliSrc << ")[:" << numModuli
     << "], dtype=jnp.uint64)\n";
  os << sum << " = " << lhsData << ".astype(jnp.uint64) + " << rhsData
     << ".astype(jnp.uint64)\n";
  os << resultName << " = jnp.where(" << sum << " >= " << moduli << ", " << sum
     << " - " << moduli << ", " << sum << ").astype(jnp.uint32)\n";
}

void JaxiteWordEmitter::emitModularReduce(StringRef targetName) {
  os << "_moduli = jnp.array(" << targetName << ".moduli, dtype=jnp.uint32)\n";
  os << targetName << ".polynomial = jnp.where(" << targetName
     << ".polynomial >= _moduli, " << targetName << ".polynomial - _moduli, "
     << targetName << ".polynomial)\n";
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
  auto pt = variableNames->getNameForValue(op.getPlaintext());
  auto result = variableNames->getNameForValue(op.getResult());
  std::string raw = result + "_raw";

  os << ctx << ".public_key = " << pk << "\n";
  os << raw << " = " << ctx << ".encrypt(" << pt << ")\n";
  emitNormalizeCiphertext(result, ctx, raw);
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(MulOp op) {
  return printMulOpHelper(
      op.getResult(), op.getLhs(), op.getRhs(), op.getCryptoContext(),
      op.getOperation(),
      [&](StringRef lhs, StringRef rhs, StringRef ctx, StringRef result,
          StringRef level) {
        std::string inputLevel = ("(" + level + ") + 1").str();
        std::string lhsWork = (result + "_lhs").str();
        std::string rhsWork = (result + "_rhs").str();
        emitNormalizeCiphertext(lhsWork, ctx, lhs, inputLevel);
        emitNormalizeCiphertext(rhsWork, ctx, rhs, inputLevel);
        os << result << " = " << ctx << ".he_mul[" << level << "].mul("
           << lhsWork << ", " << rhsWork << ")\n";
      });
}

LogicalResult JaxiteWordEmitter::printOperation(MulNoRelinOp op) {
  return printMulOpHelper(
      op.getResult(), op.getLhs(), op.getRhs(), op.getCryptoContext(),
      op.getOperation(),
      [&](StringRef lhs, StringRef rhs, StringRef ctx, StringRef result,
          StringRef level) {
        std::string inputLevel = ("(" + level + ") + 1").str();
        std::string lhsWork = (result + "_lhs").str();
        std::string rhsWork = (result + "_rhs").str();
        std::string raw = (result + "_raw").str();
        emitNormalizeCiphertext(lhsWork, ctx, lhs, inputLevel);
        emitNormalizeCiphertext(rhsWork, ctx, rhs, inputLevel);
        os << raw << " = " << ctx << ".he_mul[" << level << "].hemul_no_relin("
           << lhsWork << ", " << rhsWork << ")\n";
        emitNormalizeCiphertext(result, ctx, raw, level);
      });
}

LogicalResult JaxiteWordEmitter::printOperation(RelinOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto result = variableNames->getNameForValue(op.getOutput());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/1);
  std::string ctData = result + "_ct_data";

  os << ctData << " = " << ct << ".polynomial if hasattr(" << ct
     << ", \"polynomial\") else " << ct << "\n";
  os << result << " = " << ctx << ".he_mul[" << level << "].relinearize("
     << ctData << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(ModReduceOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto result = variableNames->getNameForValue(op.getResult());
  auto srcLevel =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);
  auto dstLevel = getCrossLevelExpr(op.getResult(), ctx, op, /*extraOffset=*/0);
  std::string ctWork = result + "_arg";

  emitNormalizeCiphertext(ctWork, ctx, ct, srcLevel);
  if (srcLevel == dstLevel) {
    emitNormalizeCiphertext(result, ctx, ctWork, dstLevel);
    return success();
  }
  os << result << " = " << ctx << ".he_rescale[" << srcLevel << ", " << dstLevel
     << "](" << ctWork << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(RotOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto result = variableNames->getNameForValue(op.getResult());
  auto rotIndex = op.getIndex();
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);
  std::string ctWork = result + "_arg";

  emitNormalizeCiphertext(ctWork, ctx, ct, level);
  os << result << " = " << ctx << ".he_rot[" << level << ", " << rotIndex
     << "].rotate(" << ctWork << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(DecryptOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto sk = variableNames->getNameForValue(op.getSecretKey());
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto result = variableNames->getNameForValue(op.getResult());
  std::string ctWork = result + "_ct";

  os << ctx << ".secret_key = " << sk << "\n";
  emitNormalizeCiphertext(ctWork, ctx, ct);
  os << "_num_moduli = " << ctWork << ".polynomial.shape[-1]\n";
  os << "_q_sub = list(getattr(" << ctWork << ", \"moduli\", " << ctx
     << ".q_towers))[:_num_moduli]\n";
  os << "_ct_for_dec = Polynomial({\"batch\": " << ctWork
     << ".polynomial.shape[0], \"num_elements\": " << ctWork
     << ".polynomial.shape[1], \"degree\": " << ctx
     << ".degree, \"precision\": 32, \"num_moduli\": _num_moduli, "
        "\"degree_layout\": ("
     << ctx << ".degree,)}, {\"moduli\": _q_sub})\n";
  os << "_ct_for_dec.set_batch_polynomial(" << ctWork << ".polynomial.reshape("
     << ctWork << ".polynomial.shape[0], " << ctWork << ".polynomial.shape[1], "
     << ctx << ".degree, _num_moduli))\n";
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
  auto result = variableNames->getNameForValue(op.getResult());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);
  std::string ctWork = result + "_arg";
  std::string ptNtt = result + "_pt_ntt";
  std::string opName = result + "_ptct";

  emitNormalizeCiphertext(ctWork, ctx, ct, level);
  os << ptNtt << " = " << pt << ".polynomial[0, 0, :, :" << ctWork
     << ".polynomial.shape[-1]].reshape(" << ctWork << ".r, " << ctWork
     << ".c, " << ctWork << ".polynomial.shape[-1]).astype(jnp.uint32)\n";
  os << opName << " = " << ctx << ".ptct_mul[" << level << "]\n";
  os << opName << ".set_plaintext(" << ptNtt << ")\n";
  os << result << " = " << opName << ".mul(" << ctWork << ", use_bat=False)\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(AddPlainOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  bool lhsPlain =
      isa<lwe::LWEPlaintextType>(getElementTypeOrSelf(op.getLhs().getType()));
  auto ct =
      variableNames->getNameForValue(lhsPlain ? op.getRhs() : op.getLhs());
  auto pt =
      variableNames->getNameForValue(lhsPlain ? op.getLhs() : op.getRhs());
  auto result = variableNames->getNameForValue(op.getResult());
  std::string numModuli = result + "_m";
  std::string ptData = result + "_pt_data";

  emitNormalizeCiphertext(result, ctx, ct);
  os << numModuli << " = " << result << ".polynomial.shape[-1]\n";
  os << ptData << " = " << pt << ".polynomial[0:1, 0:1, :, :" << numModuli
     << "].reshape(1, 1, " << result << ".r, " << result << ".c, " << numModuli
     << ")\n";
  os << "_moduli = jnp.array(" << result << ".moduli, dtype=jnp.uint32)\n";
  os << "_c0 = " << result << ".polynomial[:, 0:1, ...] + " << ptData << "\n";
  os << "_c0 = jnp.where(_c0 >= _moduli, _c0 - _moduli, _c0)\n";
  os << result << ".polynomial = " << result
     << ".polynomial.at[:, 0:1, ...].set(_c0)\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(SubPlainOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  bool lhsPlain =
      isa<lwe::LWEPlaintextType>(getElementTypeOrSelf(op.getLhs().getType()));
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  auto ct = lhsPlain ? rhs : lhs;
  auto pt = lhsPlain ? lhs : rhs;
  auto result = variableNames->getNameForValue(op.getResult());
  std::string numModuli = result + "_m";
  std::string ptData = result + "_pt_data";

  emitNormalizeCiphertext(result, ctx, ct);
  os << numModuli << " = " << result << ".polynomial.shape[-1]\n";
  os << ptData << " = " << pt << ".polynomial[0:1, 0:1, :, :" << numModuli
     << "].reshape(1, 1, " << result << ".r, " << result << ".c, " << numModuli
     << ")\n";
  os << "_moduli = jnp.array(" << result << ".moduli, dtype=jnp.uint32)\n";
  os << "_c0 = " << result << ".polynomial[:, 0:1, ...]\n";
  if (lhsPlain) {
    os << "_c0 = jnp.where(" << ptData << " < _c0, " << ptData
       << " + _moduli - _c0, " << ptData << " - _c0)\n";
    os << "_c1 = " << result << ".polynomial[:, 1:2, ...]\n";
    os << "_c1 = jnp.where(_c1 == 0, _c1, _moduli - _c1)\n";
    os << result << ".polynomial = " << result
       << ".polynomial.at[:, 1:2, ...].set(_c1)\n";
  } else {
    os << "_c0 = jnp.where(_c0 < " << ptData << ", _c0 + _moduli - " << ptData
       << ", _c0 - " << ptData << ")\n";
  }
  os << result << ".polynomial = " << result
     << ".polynomial.at[:, 0:1, ...].set(_c0)\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(AddInPlaceOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  std::string tmp = lhs + "_inplace";
  std::string rhsWork = lhs + "_rhs";

  emitNormalizeCiphertext(tmp, ctx, lhs);
  emitNormalizeCiphertext(rhsWork, ctx, rhs);
  os << tmp << ".add(" << rhsWork << ")\n";
  emitModularReduce(tmp);
  emitAssignCiphertext(lhs, tmp);
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(SubInPlaceOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  std::string tmp = lhs + "_inplace";
  std::string rhsWork = lhs + "_rhs";

  emitNormalizeCiphertext(tmp, ctx, lhs);
  emitNormalizeCiphertext(rhsWork, ctx, rhs);
  os << "_moduli = jnp.array(" << tmp << ".moduli, dtype=jnp.uint32)\n";
  os << tmp << ".polynomial = jnp.where(" << tmp << ".polynomial < " << rhsWork
     << ".polynomial, " << tmp << ".polynomial + _moduli - " << rhsWork
     << ".polynomial, " << tmp << ".polynomial - " << rhsWork
     << ".polynomial)\n";
  emitAssignCiphertext(lhs, tmp);
  return success();
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

  os << ek << "_raw = key_gen.gen_evaluation_key(" << sk << ", " << "q=" << ctx
     << ".q_towers, " << "P=" << ctx << ".p_towers, " << "dnum=" << ctx
     << ".parameters.get('dnum', 3)" << ")\n";
  os << ek << " = [\n";
  os.indent();
  os << "jnp.array(" << ek
     << "_raw[\"a\"], dtype=jnp.uint32).transpose(0, 2, 1),\n";
  os << "jnp.array(" << ek
     << "_raw[\"b\"], dtype=jnp.uint32).transpose(0, 2, 1),\n";
  os.unindent();
  os << "]\n";

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(GenRotKeyOp op) {
  auto rk = variableNames->getNameForValue(op.getResult());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto sk = variableNames->getNameForValue(op.getSecretKey());

  os << rk << " = {}\n";
  os << "for _rot_idx in [";
  llvm::interleaveComma(op.getIndices(), os);
  os << "]:\n";
  os.indent();
  os << rk << "[_rot_idx] = key_gen.gen_rotation_key(" << sk << ", " << ctx
     << ".q_towers, " << ctx << ".p_towers, rot_index=_rot_idx, dnum=" << ctx
     << ".parameters.get('dnum', 3))[_rot_idx]\n";
  os.unindent();

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(GenParamsOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());

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
  os << "\"scaling_factor\": " << op.getScalingFactor() << ",\n";

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
  auto pk = variableNames->getNameForValue(op.getPublicKey());
  auto sk = variableNames->getNameForValue(op.getSecretKey());
  auto ek = variableNames->getNameForValue(op.getEvaluationKey());

  os << ctx << ".public_key = " << pk << "\n";
  os << ctx << ".secret_key = " << sk << "\n";
  os << ctx << ".evaluation_key = " << ek << "\n";
  os << ctx << ".parameters[\"public_key\"] = " << pk << "\n";
  os << ctx << ".parameters[\"secret_key\"] = " << sk << "\n";
  os << ctx << ".parameters[\"evaluation_key\"] = " << ek << "\n";

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

  emitAssignPrefix(op.getResult());
  os << "np.zeros((";
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
