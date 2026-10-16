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
#include "lib/Dialect/JaxiteWord/IR/LevelUtils.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Target/JaxiteWord/JaxiteWordTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"              // from @llvm-project
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
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
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

static std::string getCrossLevelExpr(int current, StringRef ctxName,
                                     Operation* op, int extraOffset = 1) {
  int64_t totalOffset = getCrossLevelOffset(current, op, extraOffset);
  if (totalOffset == 0) {
    return (ctxName + ".max_level").str();
  }
  if (totalOffset == 1) {
    return (ctxName + ".max_level - 1").str();
  }
  return (ctxName + ".max_level - " + Twine(totalOffset)).str();
}

static std::string getCrossLevelExpr(Value ct, StringRef ctxName, Operation* op,
                                     int extraOffset = 1) {
  auto ctType = cast<lwe::LWECiphertextType>(ct.getType());
  int current = ctType.getModulusChain().getCurrent();
  return getCrossLevelExpr(current, ctxName, op, extraOffset);
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
          .Case<AddOp, AddPlainOp, SubOp, SquareOp, MulOp, MulNoRelinOp,
                ModReduceOp, LevelReduceOp, RotOp, RelinOp, MulPlainOp,
                AddInPlaceOp, SubInPlaceOp, EncodeOp, DecodeOp, EncryptOp,
                DecryptOp, GenParamsOp, GenKeyPairOp, GenMulKeyOp, GenRotKeyOp,
                ProgramInitializationOp, CompiledRegionOp>(
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
                arith::DivSIOp, arith::FloorDivSIOp, arith::RemSIOp,
                arith::CmpIOp, arith::SelectOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::TruncIOp>([&](auto op) { return printOperation(op); })
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

  bool hasCompiledRegion = false;
  funcOp.walk([&](CompiledRegionOp) { hasCompiledRegion = true; });
  if (hasCompiledRegion) {
    SmallVector<unsigned> ciphertextArguments;
    SmallVector<unsigned> preprocessedArguments;
    for (auto [index, argument] : llvm::enumerate(funcOp.getArguments())) {
      Type type = argument.getType();
      if (isa<CryptoContextType>(type)) continue;
      if (isa<EvalKeyType>(type)) {
        continue;
      }
      if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(type))) {
        ciphertextArguments.push_back(index);
      } else {
        preprocessedArguments.push_back(index);
      }
    }

    // Recognize the complete single-input/output graph used by the CNN demos.
    // Other signatures retain generated orchestration until explicit bindings
    // for their public container structure are available.
    SmallVector<CompiledRegionOp> regions;
    funcOp.walk([&](CompiledRegionOp region) { regions.push_back(region); });
    bool directDispatch = funcOp.getBody().hasOneBlock() &&
                          ciphertextArguments.size() == 1 &&
                          regions.size() == 1;
    bool inputIsList = false;
    bool outputIsList = false;
    auto isSingleton = [](Type type) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      return tensor && tensor.hasStaticShape() && tensor.getRank() == 1 &&
             tensor.getNumElements() == 1;
    };
    auto isZeroIndex = [](ValueRange indices) {
      if (indices.size() != 1) return false;
      auto constant = indices.front().getDefiningOp<arith::ConstantOp>();
      if (!constant) return false;
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      return integer && integer.getValue().isZero();
    };
    if (directDispatch) {
      CompiledRegionOp region = regions.front();
      SmallVector<Value> inputs;
      for (Value input : region.getInputs()) {
        if (isa<lwe::LWECiphertextType>(input.getType()))
          inputs.push_back(input);
      }
      directDispatch = inputs.size() == 1 && region.getOutputs().size() == 1;
      if (directDispatch) {
        Value input = inputs.front();
        Value publicInput = funcOp.getArgument(ciphertextArguments.front());
        if (auto extract = input.getDefiningOp<tensor::ExtractOp>()) {
          inputIsList = true;
          directDispatch = extract.getTensor() == publicInput &&
                           isSingleton(publicInput.getType()) &&
                           isZeroIndex(extract.getIndices());
        } else {
          directDispatch = input == publicInput;
        }
      }
      auto returned =
          dyn_cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());
      directDispatch &= returned && returned.getNumOperands() == 1;
      if (directDispatch) {
        Value output = returned.getOperand(0);
        if (auto insert = output.getDefiningOp<tensor::InsertOp>()) {
          outputIsList = true;
          directDispatch =
              insert.getScalar() == region.getOutputs().front() &&
              isSingleton(output.getType()) &&
              isZeroIndex(insert.getIndices()) &&
              bool(insert.getDest().getDefiningOp<tensor::EmptyOp>());
        } else {
          directDispatch = output == region.getOutputs().front();
        }
      }
      for (Operation& operation : funcOp.getBody().front()) {
        if (auto call = dyn_cast<func::CallOp>(operation)) {
          auto callee =
              funcOp->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
                  call.getCallee());
          bool deadPureCall =
              callee && !callee.isExternal() &&
              callee.getBody().hasOneBlock() &&
              llvm::all_of(call.getResults(),
                           [](Value result) { return result.use_empty(); }) &&
              llvm::all_of(callee.getBody().front(), [](Operation& inner) {
                return isMemoryEffectFree(&inner);
              });
          if (!deadPureCall) directDispatch = false;
          continue;
        }
        if (!isa<CompiledRegionOp, arith::ConstantOp, memref::LoadOp,
                 tensor::ExtractOp, tensor::InsertOp, tensor::EmptyOp,
                 func::ReturnOp>(operation)) {
          directDispatch = false;
        }
        if (auto insert = dyn_cast<tensor::InsertOp>(operation)) {
          if (!returned || returned.getNumOperands() != 1 ||
              returned.getOperand(0) != insert.getResult()) {
            directDispatch = false;
          }
        }
        if (auto extract = dyn_cast<tensor::ExtractOp>(operation)) {
          if (inputs.size() != 1 || extract.getResult() != inputs.front()) {
            directDispatch = false;
          }
        }
      }
    }

    os << "def " << funcOp.getName()
       << "__prepare_compiled(ctx, preprocessed_args, input_specs, "
          "device=None):\n";
    os.indent();
    os << "_preprocessed_args = tuple(preprocessed_args)\n";
    os << "_input_specs = tuple(input_specs)\n";
    os << "if len(_input_specs) != " << ciphertextArguments.size() << ":\n";
    os.indent();
    os << "raise ValueError(\"compiled plan input count mismatch\")\n";
    os.unindent();
    os << "if len(_preprocessed_args) != " << preprocessedArguments.size()
       << ":\n";
    os.indent();
    os << "raise ValueError(\"compiled plan preprocessing count mismatch\")\n";
    os.unindent();
    os << "def _execute(_encrypted_args):\n";
    os.indent();
    os << "return " << funcOp.getName() << "(";
    unsigned ciphertextIndex = 0;
    unsigned preprocessedIndex = 0;
    for (auto [index, argument] : llvm::enumerate(funcOp.getArguments())) {
      if (index != 0) os << ", ";
      Type type = argument.getType();
      if (isa<CryptoContextType>(type)) {
        os << "ctx";
      } else if (isa<EvalKeyType>(type)) {
        os << "{}";
      } else if (isa<lwe::LWECiphertextType>(getElementTypeOrSelf(type))) {
        os << "_encrypted_args[" << ciphertextIndex++ << "]";
      } else {
        os << "_preprocessed_args[" << preprocessedIndex++ << "]";
      }
    }
    os << ")\n";
    os.unindent();
    os << "global _compiled_prepare_only\n";
    os << "_compiled_prepare_only = True\n";
    os << "try:\n";
    os.indent();
    os << "_execute(_input_specs)\n";
    os << "raise RuntimeError(\"compiled preparation reached execution\")\n";
    os.unindent();
    os << "except _CompiledPlanPrepared as _prepared:\n";
    os.indent();
    os << "_prepared_key = _prepared.args[0]\n";
    os.unindent();
    os << "finally:\n";
    os.indent();
    os << "_compiled_prepare_only = False\n";
    os.unindent();
    if (directDispatch) {
      os << "_regions = (_compiled_region_cache[_prepared_key],)\n";
    } else {
      os << "_regions = tuple(_compiled_region_cache[key] for key in "
            "_compiled_region_cache if key[0] == \""
         << funcOp.getName() << "\" and key[2] == id(ctx))\n";
    }
    os << "return _CompiledExecutionPlan(ctx, _execute, _regions, "
          "_input_specs, (";
    for (unsigned index : ciphertextArguments) {
      auto type = cast<lwe::LWECiphertextType>(
          getElementTypeOrSelf(funcOp.getArgument(index).getType()));
      os << getCrossLevelExpr(type.getModulusChain().getCurrent(), "ctx",
                              funcOp, /*extraOffset=*/0)
         << ", ";
    }
    os << ")";
    if (directDispatch) {
      os << ", raw_input_path=(0," << (inputIsList ? " 0," : "")
         << "), raw_output_kind=\""
         << (outputIsList ? "singleton_list" : "scalar") << "\"";
    }
    os << ")\n";
    os.unindent();
    os << "\n";
    os << "def " << funcOp.getName()
       << "__execute_compiled(plan, encrypted_args):\n";
    os.indent();
    os << "if not isinstance(plan, _CompiledExecutionPlan):\n";
    os.indent();
    os << "raise TypeError(\"expected a prepared compiled execution plan\")\n";
    os.unindent();
    os << "return plan.execute(encrypted_args)\n";
    os.unindent();
    os << "\n";
  }
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
  return printBinaryOpHelper(op.getResult(), op.getLhs(), op.getRhs(),
                             op.getCryptoContext(), op, "he_add", "add");
}

LogicalResult JaxiteWordEmitter::printOperation(AddPlainOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto pt = variableNames->getNameForValue(op.getPlaintext());
  auto result = variableNames->getNameForValue(op.getOutput());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);
  os << result << " = " << ctx << ".he_add[" << level << "].add_plain(" << ct
     << ", " << pt << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(SubOp op) {
  return printBinaryOpHelper(op.getResult(), op.getLhs(), op.getRhs(),
                             op.getCryptoContext(), op, "he_sub", "sub");
}

LogicalResult JaxiteWordEmitter::printOperation(SquareOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto result = variableNames->getNameForValue(op.getResult());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/1);
  os << result << " = " << ctx << ".he_mul[" << level << "].mul(" << ct << ", "
     << ct << ")\n";
  return success();
}

void JaxiteWordEmitter::emitAssignPrefix(Value result) {
  os << variableNames->getNameForValue(result) << " = ";
}

LogicalResult JaxiteWordEmitter::printBinaryOpHelper(Value result, Value lhs,
                                                     Value rhs, Value ctx,
                                                     Operation* op,
                                                     StringRef accessor,
                                                     StringRef method) {
  auto resultName = variableNames->getNameForValue(result);
  auto lhsName = variableNames->getNameForValue(lhs);
  auto rhsName = variableNames->getNameForValue(rhs);
  auto ctxName = variableNames->getNameForValue(ctx);
  auto level = getCrossLevelExpr(lhs, ctxName, op, /*extraOffset=*/0);

  os << resultName << " = " << ctxName << "." << accessor << "[" << level
     << "]." << method << "(" << lhsName << ", " << rhsName << ")\n";
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

  callback(lhsName, rhsName, ctxName, resultName, level);
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(EncodeOp op) {
  emitAssignPrefix(op.getResult());
  auto ctxName = variableNames->getNameForValue(op.getCryptoContext());
  auto inputName = variableNames->getNameForValue(op.getInput());
  if (auto rescaleLevel = op.getRescaleLevel()) {
    auto level = getCrossLevelExpr(rescaleLevel.value(), ctxName, op,
                                   /*extraOffset=*/0);
    os << ctxName << ".encode_at_level(" << inputName << ", " << level
       << ", scale=_rescale_divisor(" << ctxName << ", " << level << "))\n";
  } else {
    os << ctxName << ".encode(" << inputName << ")\n";
  }
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(EncryptOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto pk = variableNames->getNameForValue(op.getPublicKey());
  auto pt = variableNames->getNameForValue(op.getPlaintext());
  auto result = variableNames->getNameForValue(op.getResult());
  os << ctx << ".public_key = " << pk << "\n";
  os << result << " = " << ctx << ".encrypt(" << pt << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(MulOp op) {
  return printMulOpHelper(op.getResult(), op.getLhs(), op.getRhs(),
                          op.getCryptoContext(), op.getOperation(),
                          [&](StringRef lhs, StringRef rhs, StringRef ctx,
                              StringRef result, StringRef level) {
                            os << result << " = " << ctx << ".he_mul[" << level
                               << "].mul(" << lhs << ", " << rhs << ")\n";
                          });
}

LogicalResult JaxiteWordEmitter::printOperation(MulNoRelinOp op) {
  return printMulOpHelper(op.getResult(), op.getLhs(), op.getRhs(),
                          op.getCryptoContext(), op.getOperation(),
                          [&](StringRef lhs, StringRef rhs, StringRef ctx,
                              StringRef result, StringRef level) {
                            os << result << " = " << ctx << ".he_mul[" << level
                               << "].hemul_no_relin(" << lhs << ", " << rhs
                               << ")\n";
                          });
}

LogicalResult JaxiteWordEmitter::printOperation(RelinOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto result = variableNames->getNameForValue(op.getOutput());
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/1);
  os << result << " = " << ctx << ".he_mul[" << level << "].relinearize(" << ct
     << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(ModReduceOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto result = variableNames->getNameForValue(op.getResult());
  auto srcLevel =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);
  auto dstLevel = getCrossLevelExpr(op.getResult(), ctx, op, /*extraOffset=*/0);
  if (srcLevel == dstLevel) {
    os << result << " = " << ct << "\n";
    return success();
  }
  os << result << " = " << ctx << ".he_rescale[" << srcLevel << ", " << dstLevel
     << "].rescale(" << ct << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(LevelReduceOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto result = variableNames->getNameForValue(op.getResult());
  auto srcLevel =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);
  auto dstLevel = getCrossLevelExpr(op.getResult(), ctx, op, /*extraOffset=*/0);
  if (srcLevel == dstLevel) {
    os << result << " = " << ct << "\n";
    return success();
  }
  os << result << " = " << ctx << ".he_level_reduce[" << srcLevel << ", "
     << dstLevel << "].level_reduce(" << ct << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(RotOp op) {
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto result = variableNames->getNameForValue(op.getResult());
  auto rotIndex = op.getIndex();
  auto level =
      getCrossLevelExpr(op.getCiphertext(), ctx, op, /*extraOffset=*/0);
  os << result << " = " << ctx << ".he_rot[" << level << ", " << rotIndex
     << "].rotate(" << ct << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(DecryptOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto sk = variableNames->getNameForValue(op.getSecretKey());
  auto ct = variableNames->getNameForValue(op.getCiphertext());
  auto result = variableNames->getNameForValue(op.getResult());
  os << ctx << ".secret_key = " << sk << "\n";
  emitAssignPrefix(op.getResult());
  os << ctx << ".decrypt(" << ct << ")\n";
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
  os << result << " = " << ctx << ".ptct_mul[" << level << "].mul(" << ct
     << ", " << pt << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(AddInPlaceOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  auto level = getCrossLevelExpr(op.getLhs(), ctx, op, /*extraOffset=*/0);
  os << lhs << " = " << ctx << ".he_add[" << level << "].add(" << lhs << ", "
     << rhs << ")\n";
  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(SubInPlaceOp op) {
  auto ctx = variableNames->getNameForValue(op.getCryptoContext());
  auto lhs = variableNames->getNameForValue(op.getLhs());
  auto rhs = variableNames->getNameForValue(op.getRhs());
  auto level = getCrossLevelExpr(op.getLhs(), ctx, op, /*extraOffset=*/0);
  os << lhs << " = " << ctx << ".he_sub[" << level << "].sub(" << lhs << ", "
     << rhs << ")\n";
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
  os << "\"scaling_factor\": " << op.getScalingFactor() << ",\n";
  os << "\"output_scale\": " << op.getScalingFactor() << ",\n";

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
  os << "total_rotation_indices=[";
  auto rotIndices = op.getTotalRotationIndices();
  llvm::interleaveComma(rotIndices, os);
  os << "], ";

  os << "dnum=" << op.getDnum() << ", ";
  os << "r=" << op.getR() << ", ";
  os << "c=" << op.getC() << ", ";
  os << "batch=" << op.getBatch();
  os << ", cache_rotation_keys=False";
  os << ")\n";

  auto requirementIndices = op.getRotationRequirementIndices();
  auto levelOffsets = op.getRotationLevelOffsets();
  for (auto [levelOffset, rotIndex] :
       llvm::zip(levelOffsets, requirementIndices)) {
    os << ctx << ".he_rot[" << ctx << ".max_level";
    if (levelOffset != 0) os << " - " << levelOffset;
    os << ", " << rotIndex << "]\n";
  }

  return success();
}

LogicalResult JaxiteWordEmitter::printOperation(CompiledRegionOp op) {
  Block& body = op.getBody().front();
  auto yield = cast<CompiledRegionYieldOp>(body.getTerminator());
  int64_t regionId = op.getRegionId();

  DenseMap<Value, std::string> outerNames;
  DenseMap<Value, std::string> rawNames;
  Value contextInput;
  Value templateInput;
  unsigned nextValue = 0;
  for (auto [input, argument] :
       llvm::zip(op.getInputs(), body.getArguments())) {
    std::string outer = variableNames->getNameForValue(input);
    outerNames[argument] = outer;
    if (isa<CryptoContextType>(argument.getType())) {
      contextInput = argument;
      rawNames[argument] = outer;
      continue;
    }
    if (isa<lwe::LWECiphertextType>(argument.getType()) && !templateInput) {
      templateInput = argument;
    }
    rawNames[argument] =
        (Twine("_region_") + Twine(regionId) + "_arg_" + Twine(nextValue++))
            .str();
  }
  if (!contextInput || !templateInput) {
    return op.emitOpError(
        "compiled region requires context and ciphertext inputs");
  }
  std::string ctx = outerNames.lookup(contextInput);

  DenseMap<Value, std::string> scaleExpressions;
  std::function<std::string(Value)> scaleOf = [&](Value value) -> std::string {
    auto found = scaleExpressions.find(value);
    if (found != scaleExpressions.end()) return found->second;
    std::string expression;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      expression = (Twine("getattr(") + outerNames.lookup(argument) +
                    ", \"_ckks_scale\", None)")
                       .str();
    } else if (auto mul = value.getDefiningOp<MulPlainOp>()) {
      expression =
          (Twine("_compiled_scale_mul(") + scaleOf(mul.getCiphertext()) +
           ", getattr(" + outerNames.lookup(mul.getPlaintext()) +
           ", \"_ckks_scale\", None))")
              .str();
    } else if (auto add = value.getDefiningOp<AddOp>()) {
      expression = scaleOf(add.getLhs());
    } else if (auto mul = value.getDefiningOp<MulNoRelinOp>()) {
      expression = "_compiled_scale_mul(" + scaleOf(mul.getLhs()) + ", " +
                   scaleOf(mul.getRhs()) + ")";
    } else if (auto relin = value.getDefiningOp<RelinOp>()) {
      expression = scaleOf(relin.getCiphertext());
    } else if (auto mul = value.getDefiningOp<MulOp>()) {
      expression = "_compiled_scale_rescale(_compiled_scale_mul(" +
                   scaleOf(mul.getLhs()) + ", " + scaleOf(mul.getRhs()) +
                   ", \"jaxiteword.mul\", " +
                   getCrossLevelExpr(mul.getLhs(), ctx, mul, 0) + "), " + ctx +
                   ", " + getCrossLevelExpr(mul.getLhs(), ctx, mul, 0) + ", " +
                   getCrossLevelExpr(value, ctx, mul, 0) + ")";
    } else if (auto square = value.getDefiningOp<SquareOp>()) {
      expression = "_compiled_scale_rescale(_compiled_scale_mul(" +
                   scaleOf(square.getCiphertext()) + ", " +
                   scaleOf(square.getCiphertext()) +
                   ", \"jaxiteword.square\", " +
                   getCrossLevelExpr(square.getCiphertext(), ctx, square, 0) +
                   "), " + ctx + ", " +
                   getCrossLevelExpr(square.getCiphertext(), ctx, square, 0) +
                   ", " + getCrossLevelExpr(value, ctx, square, 0) + ")";
    } else if (auto addPlain = value.getDefiningOp<AddPlainOp>()) {
      expression = scaleOf(addPlain.getCiphertext());
    } else if (auto rot = value.getDefiningOp<RotOp>()) {
      expression = scaleOf(rot.getCiphertext());
    } else if (auto reduce = value.getDefiningOp<ModReduceOp>()) {
      std::string src = getCrossLevelExpr(reduce.getCiphertext(), ctx, reduce,
                                          /*extraOffset=*/0);
      std::string dst = getCrossLevelExpr(reduce.getResult(), ctx, reduce,
                                          /*extraOffset=*/0);
      expression =
          (Twine("_compiled_scale_rescale(") + scaleOf(reduce.getCiphertext()) +
           ", " + ctx + ", " + src + ", " + dst + ")")
              .str();
    } else if (auto reduce = value.getDefiningOp<LevelReduceOp>()) {
      expression = scaleOf(reduce.getCiphertext());
    } else {
      expression = "None";
    }
    if (Operation* defining = value.getDefiningOp()) {
      if (isa<MulPlainOp, MulNoRelinOp>(defining)) {
        expression.pop_back();
        expression += ", \"" + defining->getName().getStringRef().str() +
                      "\", " + getCrossLevelExpr(value, ctx, defining, 0) + ")";
      }
    }
    scaleExpressions[value] = expression;
    return expression;
  };

  DenseMap<Value, std::string> nsdExpressions;
  std::function<std::string(Value)> nsdOf = [&](Value value) -> std::string {
    auto found = nsdExpressions.find(value);
    if (found != nsdExpressions.end()) return found->second;
    std::string expression;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      expression = (Twine("getattr(") + outerNames.lookup(argument) +
                    ", \"_ckks_nsd\", 1)")
                       .str();
    } else if (auto mul = value.getDefiningOp<MulPlainOp>()) {
      expression = (Twine("(") + nsdOf(mul.getCiphertext()) + " + 1)").str();
    } else if (auto mul = value.getDefiningOp<MulNoRelinOp>()) {
      expression =
          "(" + nsdOf(mul.getLhs()) + " + " + nsdOf(mul.getRhs()) + ")";
    } else if (isa_and_nonnull<MulOp, SquareOp>(value.getDefiningOp())) {
      expression = "1";
    } else if (Operation* defining = value.getDefiningOp()) {
      Value source;
      if (auto add = dyn_cast<AddOp>(defining)) source = add.getLhs();
      if (auto addPlain = dyn_cast<AddPlainOp>(defining))
        source = addPlain.getCiphertext();
      if (auto rot = dyn_cast<RotOp>(defining)) source = rot.getCiphertext();
      if (auto reduce = dyn_cast<ModReduceOp>(defining))
        source = reduce.getCiphertext();
      if (auto reduce = dyn_cast<LevelReduceOp>(defining))
        source = reduce.getCiphertext();
      if (auto relin = dyn_cast<RelinOp>(defining))
        source = relin.getCiphertext();
      expression = source ? nsdOf(source) : "1";
    } else {
      expression = "1";
    }
    nsdExpressions[value] = expression;
    return expression;
  };

  DenseMap<Value, int64_t> elementCounts;
  std::function<FailureOr<int64_t>(Value)> elementsOf =
      [&](Value value) -> FailureOr<int64_t> {
    auto found = elementCounts.find(value);
    if (found != elementCounts.end()) return found->second;
    int64_t elements = 2;
    if (auto mul = value.getDefiningOp<MulNoRelinOp>()) {
      auto lhs = elementsOf(mul.getLhs());
      auto rhs = elementsOf(mul.getRhs());
      if (failed(lhs) || failed(rhs) || *lhs != 2 || *rhs != 2) {
        return failure();
      }
      elements = 3;
    } else if (auto relin = value.getDefiningOp<RelinOp>()) {
      auto input = elementsOf(relin.getCiphertext());
      if (failed(input) || *input != 3) return failure();
      elements = 2;
    } else if (isa_and_nonnull<MulOp, SquareOp>(value.getDefiningOp())) {
      elements = 2;
    } else if (Operation* defining = value.getDefiningOp()) {
      Value source;
      if (auto mul = dyn_cast<MulPlainOp>(defining))
        source = mul.getCiphertext();
      if (auto add = dyn_cast<AddOp>(defining)) source = add.getLhs();
      if (auto addPlain = dyn_cast<AddPlainOp>(defining))
        source = addPlain.getCiphertext();
      if (auto rot = dyn_cast<RotOp>(defining)) source = rot.getCiphertext();
      if (auto reduce = dyn_cast<ModReduceOp>(defining))
        source = reduce.getCiphertext();
      if (auto reduce = dyn_cast<LevelReduceOp>(defining))
        source = reduce.getCiphertext();
      if (source) {
        auto input = elementsOf(source);
        if (failed(input)) return failure();
        elements = *input;
      }
    }
    elementCounts[value] = elements;
    return elements;
  };

  // Bind every evaluator outside the traced function. Large rotation keys and
  // plaintexts are explicit dynamic arguments rather than XLA constants.
  std::string cacheKey =
      (Twine("_region_") + Twine(regionId) + "_cache_key").str();
  os << cacheKey << " = (\"" << op->getParentOfType<func::FuncOp>().getName()
     << "\", " << regionId << ", id(" << ctx << ")";
  for (BlockArgument argument : body.getArguments()) {
    if (isa<CryptoContextType, EvalKeyType>(argument.getType())) continue;
    if (isa<lwe::LWECiphertextType>(argument.getType())) {
      os << ", _compiled_ciphertext_fingerprint(" << ctx << ", "
         << outerNames.lookup(argument) << ", "
         << getCrossLevelExpr(argument, ctx, op, /*extraOffset=*/0) << ")";
    } else {
      os << ", _compiled_plaintext_fingerprint(" << outerNames.lookup(argument)
         << ")";
    }
  }
  os << ")\n";
  os << "if " << cacheKey << " not in _compiled_region_cache:\n";
  os.indent();

  DenseMap<Operation*, std::string> evaluators;
  DenseMap<Operation*, std::string> preparedPlaintexts;
  SmallVector<std::string> runtimeArguments;
  SmallVector<std::string> staticArgumentNames;
  SmallVector<std::string> callArguments;
  SmallVector<bool> staticCallArguments;
  for (BlockArgument argument : body.getArguments()) {
    if (isa<CryptoContextType, EvalKeyType>(argument.getType())) continue;
    std::string raw = rawNames.lookup(argument);
    if (isa<lwe::LWECiphertextType>(argument.getType())) {
      runtimeArguments.push_back(raw);
      callArguments.push_back(outerNames.lookup(argument) + ".polynomial");
      staticCallArguments.push_back(false);
      continue;
    }
    if (argument.use_empty()) {
      return op.emitOpError("unused non-ciphertext compiled-region input");
    }
    for (Operation* user : argument.getUsers()) {
      if (!isa<MulPlainOp, AddPlainOp>(user)) {
        return op.emitOpError("plaintext region input has an unsupported user");
      }
      Value ciphertext = isa<MulPlainOp>(user)
                             ? cast<MulPlainOp>(user).getCiphertext()
                             : cast<AddPlainOp>(user).getCiphertext();
      std::string level =
          getCrossLevelExpr(ciphertext, ctx, user, /*extraOffset=*/0);
      std::string prepared = (Twine("_region_") + Twine(regionId) +
                              "_prepared_" + Twine(preparedPlaintexts.size()))
                                 .str();
      preparedPlaintexts[user] = prepared;
      os << prepared << " = ";
      if (isa<AddPlainOp>(user)) {
        os << "_prepare_compiled_add_plain(" << ctx << ", "
           << outerNames.lookup(argument) << ", " << level << ", "
           << scaleOf(ciphertext) << ")\n";
      } else {
        os << "_prepare_compiled_plaintext(" << ctx << ", "
           << outerNames.lookup(argument) << ", " << level << ")\n";
      }
      staticArgumentNames.push_back(prepared);
      callArguments.push_back(prepared);
      staticCallArguments.push_back(true);
    }
  }

  unsigned operationIndex = 0;
  for (Operation& operation : body.without_terminator()) {
    std::string evaluator = (Twine("_region_") + Twine(regionId) +
                             "_evaluator_" + Twine(operationIndex++))
                                .str();
    evaluators[&operation] = evaluator;
    if (auto mul = dyn_cast<MulPlainOp>(operation)) {
      os << evaluator << " = " << ctx << ".ptct_mul["
         << getCrossLevelExpr(mul.getCiphertext(), ctx, mul,
                              /*extraOffset=*/0)
         << "]\n";
    } else if (auto add = dyn_cast<AddOp>(operation)) {
      os << "_compiled_validate_add(" << scaleOf(add.getLhs()) << ", "
         << nsdOf(add.getLhs()) << ", " << scaleOf(add.getRhs()) << ", "
         << nsdOf(add.getRhs()) << ")\n";
      os << evaluator << " = " << ctx << ".he_add["
         << getCrossLevelExpr(add.getLhs(), ctx, add, /*extraOffset=*/0)
         << "]\n";
    } else if (auto addPlain = dyn_cast<AddPlainOp>(operation)) {
      os << evaluator << " = " << ctx << ".he_add["
         << getCrossLevelExpr(addPlain.getCiphertext(), ctx, addPlain,
                              /*extraOffset=*/0)
         << "]\n";
    } else if (auto rot = dyn_cast<RotOp>(operation)) {
      os << evaluator << " = " << ctx << ".he_rot["
         << getCrossLevelExpr(rot.getCiphertext(), ctx, rot,
                              /*extraOffset=*/0)
         << ", " << rot.getIndex() << "]\n";
      for (unsigned i = 0; i < 3; ++i) {
        std::string state = evaluator + "_state_" + std::to_string(i);
        staticArgumentNames.push_back(state);
        callArguments.push_back(evaluator + "._rotation_state()[" +
                                std::to_string(i) + "]");
        staticCallArguments.push_back(true);
      }
    } else if (isa<MulNoRelinOp, RelinOp, MulOp, SquareOp>(operation)) {
      Value input = operation.getOperand(1);
      os << evaluator << " = " << ctx << ".he_mul["
         << getCrossLevelExpr(input, ctx, &operation, 1) << "]\n";
    } else if (auto reduce = dyn_cast<ModReduceOp>(operation)) {
      os << evaluator << " = " << ctx << ".he_rescale["
         << getCrossLevelExpr(reduce.getCiphertext(), ctx, reduce,
                              /*extraOffset=*/0)
         << ", "
         << getCrossLevelExpr(reduce.getResult(), ctx, reduce,
                              /*extraOffset=*/0)
         << "]\n";
    } else if (auto reduce = dyn_cast<LevelReduceOp>(operation)) {
      os << evaluator << " = " << ctx << ".he_level_reduce["
         << getCrossLevelExpr(reduce.getCiphertext(), ctx, reduce,
                              /*extraOffset=*/0)
         << ", "
         << getCrossLevelExpr(reduce.getResult(), ctx, reduce,
                              /*extraOffset=*/0)
         << "]\n";
    }
  }

  std::string functionName =
      (Twine("_compiled_region_") + Twine(regionId) + "_raw").str();
  os << "def " << functionName << "(";
  llvm::interleaveComma(runtimeArguments, os);
  if (!runtimeArguments.empty()) os << ", ";
  os << "_region_" << regionId << "_static_args_in";
  os << "):\n";
  os.indent();
  for (auto [index, name] : llvm::enumerate(staticArgumentNames)) {
    os << name << " = _region_" << regionId << "_static_args_in[" << index
       << "]\n";
  }
  for (Operation& operation : body.without_terminator()) {
    std::string result =
        (Twine("_region_") + Twine(regionId) + "_value_" + Twine(nextValue++))
            .str();
    rawNames[operation.getResult(0)] = result;
    std::string evaluator = evaluators.lookup(&operation);
    if (auto mul = dyn_cast<MulPlainOp>(operation)) {
      os << result << " = " << evaluator << "._mul_array("
         << rawNames.lookup(mul.getCiphertext()) << ", "
         << preparedPlaintexts.lookup(&operation) << ")\n";
    } else if (auto add = dyn_cast<AddOp>(operation)) {
      os << result << " = " << evaluator << "._add_array("
         << rawNames.lookup(add.getLhs()) << ", "
         << rawNames.lookup(add.getRhs()) << ")\n";
    } else if (auto addPlain = dyn_cast<AddPlainOp>(operation)) {
      os << result << " = " << evaluator << "._add_plain_array("
         << rawNames.lookup(addPlain.getCiphertext()) << ", "
         << preparedPlaintexts.lookup(&operation) << ")\n";
    } else if (auto rot = dyn_cast<RotOp>(operation)) {
      os << result << " = " << evaluator << "._rotate_array("
         << rawNames.lookup(rot.getCiphertext());
      for (unsigned i = 0; i < 3; ++i)
        os << ", " << evaluator << "_state_" << i;
      os << ")\n";
    } else if (auto mul = dyn_cast<MulNoRelinOp>(operation)) {
      os << result << " = " << evaluator
         << "._hemul._hemul_no_relin_array(jnp.concatenate(("
         << rawNames.lookup(mul.getLhs()) << ", "
         << rawNames.lookup(mul.getRhs()) << "), axis=1))\n";
    } else if (auto relin = dyn_cast<RelinOp>(operation)) {
      os << result << " = " << evaluator << "._hemul._relinearize_array("
         << rawNames.lookup(relin.getCiphertext()) << ")\n";
    } else if (auto mul = dyn_cast<MulOp>(operation)) {
      os << result << " = " << evaluator << "._mul_array("
         << rawNames.lookup(mul.getLhs()) << ", "
         << rawNames.lookup(mul.getRhs()) << ")\n";
    } else if (auto square = dyn_cast<SquareOp>(operation)) {
      os << result << " = " << evaluator << "._square_array("
         << rawNames.lookup(square.getCiphertext()) << ")\n";
    } else if (auto reduce = dyn_cast<ModReduceOp>(operation)) {
      os << result << " = " << evaluator << "._rescale_array("
         << rawNames.lookup(reduce.getCiphertext()) << ")\n";
    } else if (auto reduce = dyn_cast<LevelReduceOp>(operation)) {
      os << result << " = " << evaluator << "._level_reduce_array("
         << rawNames.lookup(reduce.getCiphertext()) << ")\n";
    }
  }
  os << "return (";
  llvm::interleaveComma(yield.getValues(), os,
                        [&](Value value) { os << rawNames.lookup(value); });
  if (yield.getValues().size() == 1) os << ",";
  os << ")\n";
  os.unindent();

  std::string jitDeviceArgument;
  for (auto [argument, isStatic] :
       llvm::zip(callArguments, staticCallArguments)) {
    if (!isStatic) {
      jitDeviceArgument = argument;
      break;
    }
  }
  if (jitDeviceArgument.empty()) {
    return op.emitOpError("compiled region requires a runtime array input");
  }
  os << "_compiled_region_cache[" << cacheKey << "] = (jax.jit(" << functionName
     << ", device=_compiled_array_device(" << jitDeviceArgument << ")).lower(";
  bool emittedRuntime = false;
  for (auto [argument, isStatic] :
       llvm::zip(callArguments, staticCallArguments)) {
    if (isStatic) continue;
    if (emittedRuntime) os << ", ";
    os << argument;
    emittedRuntime = true;
  }
  if (emittedRuntime) os << ", ";
  os << "(";
  bool emittedStatic = false;
  for (auto [argument, isStatic] :
       llvm::zip(callArguments, staticCallArguments)) {
    if (!isStatic) continue;
    if (emittedStatic) os << ", ";
    os << argument;
    emittedStatic = true;
  }
  if (emittedStatic) os << ",";
  os << ")).compile(), (";
  emittedStatic = false;
  for (auto [argument, isStatic] :
       llvm::zip(callArguments, staticCallArguments)) {
    if (!isStatic) continue;
    if (emittedStatic) os << ", ";
    os << argument;
    emittedStatic = true;
  }
  if (emittedStatic) os << ",";
  os << "), (";
  for (auto [index, output] : llvm::enumerate(op.getOutputs())) {
    Value yielded = yield.getValues()[index];
    auto elements = elementsOf(yielded);
    if (failed(elements)) {
      return op.emitOpError(
          "compiled region has incompatible ciphertext element counts");
    }
    if (index != 0) os << ", ";
    os << "_prepare_compiled_output_template(" << ctx << ", "
       << outerNames.lookup(templateInput) << ", "
       << getCrossLevelExpr(output, ctx, op, /*extraOffset=*/0) << ", "
       << *elements << ", " << scaleOf(yielded) << ", " << nsdOf(yielded)
       << ")";
  }
  if (op.getOutputs().size() == 1) os << ",";
  os << "))\n";
  os.unindent();
  os << "_region_" << regionId << "_compiled, _region_" << regionId
     << "_static_args, _region_" << regionId
     << "_output_templates = _compiled_region_cache[" << cacheKey << "]\n";
  os << "if _compiled_prepare_only:\n";
  os.indent();
  os << "raise _CompiledPlanPrepared(" << cacheKey << ")\n";
  os.unindent();
  os << "_region_" << regionId << "_outputs = _region_" << regionId
     << "_compiled(";
  emittedRuntime = false;
  for (auto [argument, isStatic] :
       llvm::zip(callArguments, staticCallArguments)) {
    if (isStatic) continue;
    if (emittedRuntime) os << ", ";
    os << argument;
    emittedRuntime = true;
  }
  if (emittedRuntime) os << ", ";
  os << "_region_" << regionId << "_static_args";
  os << ")\n";

  for (auto [index, output] : llvm::enumerate(op.getOutputs())) {
    os << variableNames->getNameForValue(output) << " = _region_" << regionId
       << "_output_templates[" << index << "]._clone_with_payload(_region_"
       << regionId << "_outputs[" << index << "])\n";
  }
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
  auto blockEmitsStatement = [&](Block& block) {
    return llvm::any_of(block, [&](Operation& nested) {
      auto yield = dyn_cast<scf::YieldOp>(nested);
      if (!yield) return true;
      return llvm::any_of(op.getResultTypes(),
                          [](Type type) { return !isa<ShapedType>(type); });
    });
  };
  if (!blockEmitsStatement(*op.thenBlock())) os << "pass\n";
  for (Operation& thenOp : *op.thenBlock()) {
    if (failed(translate(thenOp))) {
      return thenOp.emitOpError() << "Failed to translate if then block";
    }
  }
  os.unindent();

  bool hasElseStatement = blockEmitsStatement(*op.elseBlock());
  if (hasElseStatement) {
    os << "else:\n";
    os.indent();
    for (Operation& elseOp : *op.elseBlock()) {
      if (failed(translate(elseOp))) {
        return elseOp.emitOpError() << "Failed to translate if else block";
      }
    }
    os.unindent();
  }

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

LogicalResult JaxiteWordEmitter::printOperation(arith::FloorDivSIOp op) {
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
          [&](auto) { return std::string("Polynomial"); })
      .Case<PublicKeyType>([&](auto) { return std::string("np.ndarray"); })
      .Case<PrivateKeyType>([&](auto) { return std::string("np.ndarray"); })
      .Case<EvalKeyType>([&](auto) { return std::string("dict"); })
      .Case<CryptoContextType>(
          [&](auto) { return std::string("ckks.CKKSContext"); })
      .Case<lwe::LWEPlaintextType>(
          [&](auto) { return std::string("Polynomial"); })
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
