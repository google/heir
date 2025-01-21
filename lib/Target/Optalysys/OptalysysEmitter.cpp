#include "lib/Target/Optalysys/OptalysysEmitter.h"

#include <string>
#include <string_view>

#include "lib/Analysis/SelectVariableNames/SelectVariableNames.h"
#include "lib/Dialect/Optalysys/IR/OptalysysDialect.h"
#include "lib/Dialect/Optalysys/IR/OptalysysOps.h"
#include "lib/Dialect/Optalysys/IR/OptalysysTypes.h"
#include "lib/Target/Optalysys/OptalysysTemplates.h"
#include "lib/Utils/TargetUtils.h"
#include "llvm/include/llvm/ADT/StringExtras.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/CommandLine.h"      // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"   // from @llvm-project
#include "llvm/include/llvm/Support/ManagedStatic.h"    // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/EmitC/IR/EmitC.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"       // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace optalysys {

namespace {

FailureOr<std::string> convertType(Type type, Location loc) {
  return llvm::TypeSwitch<Type, FailureOr<std::string>>(type)
      .Case<IntegerType>([&](auto ty) -> FailureOr<std::string> {
        auto width = ty.getWidth();
        if (width == 1) {
          return std::string("bool");
        }
        if (width != 8 && width != 16 && width != 32 && width != 64) {
          return failure();
        }
        return std::string(llvm::formatv("int{0}_t", width));
      })
      .Case<emitc::PointerType>([&](auto ty) -> FailureOr<std::string> {
        auto pointeeTy = convertType(ty.getPointee(), loc);
        if (failed(pointeeTy)) {
          return failure();
        }
        return pointeeTy.value() + std::string("*");
      })
      .Case<LutFunctionType>([&](auto ty) { return std::string("lut_fn"); })
      .Case<CiphertextSizeTType>([&](auto ty) { return std::string("size_t"); })
      .Case<emitc::OpaqueType>(
          [&](emitc::OpaqueType ty) { return std::string(ty.getValue()); })
      .Default([&](Type) -> FailureOr<std::string> { return failure(); });
}

}  // namespace

LogicalResult translateToOptalysys(Operation *op, llvm::raw_ostream &os) {
  SelectVariableNames variableNames(op);
  OptalysysEmitter emitter(os, &variableNames);
  LogicalResult result = emitter.translate(*op);
  return result;
}

LogicalResult OptalysysEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops
          .Case<func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Arith ops
          .Case<arith::ConstantOp>([&](auto op) { return printOperation(op); })
          // Optalysys ops
          .Case<RlweGenLutOp, DeviceMultiCMux1024PbsBOp, TrivialLweEncryptOp,
                MulScalarOp, AddOp, LweCreateBatchOp, LoadOp>(
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

LogicalResult OptalysysEmitter::printOperation(ModuleOp moduleOp) {
  os << kModulePreludeTemplate;

  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult OptalysysEmitter::printOperation(func::FuncOp funcOp) {
  if (funcOp.getNumResults() != 1) {
    return emitError(funcOp.getLoc(),
                     llvm::formatv("Only functions with a single return type "
                                   "are supported, but this function has ",
                                   funcOp.getNumResults()));
    return failure();
  }

  Type result = funcOp.getResultTypes()[0];
  if (failed(emitType(result, funcOp->getLoc()))) {
    return emitError(funcOp.getLoc(),
                     llvm::formatv("Failed to emit type {0}", result));
  }

  os << " " << funcOp.getName() << "(";

  for (Value arg : funcOp.getArguments()) {
    if (failed(convertType(arg.getType(), arg.getLoc()))) {
      return emitError(funcOp.getLoc(),
                       llvm::formatv("Failed to emit type {0}", arg.getType()));
    }
  }
  os << commaSeparatedValues(funcOp.getArguments(), [&](Value value) {
    auto res = convertType(value.getType(), funcOp->getLoc());
    return res.value() + " " + variableNames->getNameForValue(value);
  });

  os << ") {\n";
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

LogicalResult OptalysysEmitter::printOperation(func::ReturnOp op) {
  if (op.getNumOperands() != 1) {
    return emitError(op.getLoc(), "Only one return value supported");
  }
  os << "return " << variableNames->getNameForValue(op.getOperands()[0])
     << ";\n";
  return success();
}

LogicalResult OptalysysEmitter::printOperation(arith::ConstantOp op) {
  if (isa<IndexType>(op.getResult().getType())) {
    return success();
  }
  auto valueAttr = op.getValue();
  auto res =
      llvm::TypeSwitch<Attribute, LogicalResult>(valueAttr)
          .Case<IntegerAttr>([&](IntegerAttr intAttr) {
            // Constant integers may be unused if their uses directly output the
            // constant value (e.g. tensor.insert and tensor.extract use the
            // defining constant values of indices if available).
            os << "[[maybe_unused]] ";
            if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc()))) {
              return failure();
            }
            os << intAttr.getValue() << ";\n";
            return success();
          })
          .Default([&](auto) { return failure(); });
  if (failed(res)) {
    return res;
  }
  return success();
}

LogicalResult OptalysysEmitter::printOperation(RlweGenLutOp op) {
  // const size_t index_lut = rlwe_gen_lut(f, &Parameters);
  os << "const size_t " << variableNames->getNameForValue(op.getResult())
     << " = ";
  auto lutLambda = llvm::formatv(
      "[](uint_t x) -> uint_t { return static_cast<uint_t>(({0} >> "
      "static_cast<uint8_t>(x)) & 1); }",
      op.getLut());
  os << "rlwe_gen_lut(" << lutLambda << ", "
     << variableNames->getNameForValue(op.getParameters()) << ");\n";
  return success();
}

LogicalResult OptalysysEmitter::printOperation(DeviceMultiCMux1024PbsBOp op) {
  // size_t v13 = device_multi_cmux_1024_pbs_b(v9, 1, v12);
  if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc()))) {
    return failure();
  }
  os << "device_multi_cmux_1024_pbs_b("
     << variableNames->getNameForValue(op.getInput()) << ", "
     << op.getBatchSize() << ", " << variableNames->getNameForValue(op.getF())
     << ");\n";
  return success();
}

LogicalResult OptalysysEmitter::printOperation(TrivialLweEncryptOp op) {
  // size_t v13 = trivial_lwe_encrypt(message, params);
  if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc()))) {
    return failure();
  }
  os << "trivial_lwe_encrypt(" << variableNames->getNameForValue(op.getInput())
     << ", " << variableNames->getNameForValue(op.getParameters()) << ");\n";
  return success();
}

LogicalResult OptalysysEmitter::printOperation(MulScalarOp op) {
  // size_t v13 = lwe_ops_mul_scalar(message, scalar);
  if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc()))) {
    return failure();
  }
  os << "lwe_ops_mul_scalar("
     << variableNames->getNameForValue(op.getCiphertext()) << ", "
     << variableNames->getNameForValue(op.getScalar()) << ");\n";
  return success();
}

LogicalResult OptalysysEmitter::printOperation(AddOp op) {
  // size_t v13 = lwe_ops_add(c1, c2);
  if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc()))) {
    return failure();
  }
  os << "lwe_ops_add(" << variableNames->getNameForValue(op.getLhs()) << ", "
     << variableNames->getNameForValue(op.getRhs()) << ");\n";
  return success();
}

LogicalResult OptalysysEmitter::printOperation(LweCreateBatchOp op) {
  // size_t v13 = lwe_create_batch(const size_t* const indices, const size_t
  // batch_size)
  if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc()))) {
    return failure();
  }
  os << "lwe_create_batch(";
  os << "{";
  os << commaSeparatedValues(op.getInput(), [&](Value value) {
    return variableNames->getNameForValue(value);
  });
  os << "}";
  os << ", " << op.getSize() << ");\n";
  return success();
}

LogicalResult OptalysysEmitter::printOperation(LoadOp op) {
  // size_t v13 = c1 + x;
  if (failed(emitTypedAssignPrefix(op.getResult(), op.getLoc()))) {
    return failure();
  }
  os << variableNames->getNameForValue(op.getInput()) << " + ";
  if (auto constantOp =
          dyn_cast_or_null<arith::ConstantOp>(op.getIndex().getDefiningOp())) {
    auto valueAttr = constantOp.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
      os << intAttr.getInt();
    }
  } else {
    os << variableNames->getNameForValue(op.getIndex());
  }
  os << ";\n";
  return success();
}

LogicalResult OptalysysEmitter::emitTypedAssignPrefix(Value result,
                                                      Location loc) {
  if (failed(emitType(result.getType(), loc))) {
    return failure();
  }
  os << " " << variableNames->getNameForValue(result) << " = ";
  return success();
}

LogicalResult OptalysysEmitter::emitType(Type type, Location loc) {
  auto result = convertType(type, loc);
  if (failed(result)) {
    return failure();
  }
  os << result;
  return success();
}

OptalysysEmitter::OptalysysEmitter(raw_ostream &os,
                                   SelectVariableNames *variableNames)
    : os(os), variableNames(variableNames) {}

void registerToOptalysysTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-optalysys",
      "translate the optalysys dialect to C code against the Optalysys API",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToOptalysys(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, func::FuncDialect,
                        optalysys::OptalysysDialect, emitc::EmitCDialect>();
      });
}

}  // namespace optalysys
}  // namespace heir
}  // namespace mlir
