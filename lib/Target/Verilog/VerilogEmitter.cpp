#include "lib/Target/Verilog/VerilogEmitter.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "lib/Conversion/MemrefToArith/Utils.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "lib/Target/Utils.h"
#include "llvm/include/llvm/ADT/STLExtras.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallString.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"         // from @llvm-project
#include "llvm/include/llvm/ADT/StringExtras.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/ilist.h"               // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"   // from @llvm-project
#include "llvm/include/llvm/Support/FormatVariadic.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Utils.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
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

namespace {

static constexpr std::string_view kOutputPrefix = "_out_";

bool shouldMapToSigned(IntegerType::SignednessSemantics val) {
  switch (val) {
    case IntegerType::Signless:
    case IntegerType::Signed:
      return true;
    case IntegerType::Unsigned:
      return false;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

// wireDeclaration returns a string declaring a verilog wire, for e.g.
//   wire [signed] <NAME> [<SIZE> - 1: 0];
std::string wireDeclaration(IntegerType iType, int32_t width) {
  if (width == 1) {
    return "wire";
  }
  // Ensure that we add a signed modifier to wire declarations holding signed
  // integers in order to signal to verilog to use signed operations. All
  // verilog operations are unsigned by default, and verilog requires either
  // defining both wires of an operation with the signed modifier, or else use
  // the builtin $signed() function around both operands to use a signed
  // operations. Fortunately, MLIR requires that both operands of a signed
  // operations, like comparisons (cmpi), are the same type, so if we encounter
  // a cmpi, both its wire declarations will have this signed modifier.
  std::string_view signedModifier =
      shouldMapToSigned(iType.getSignedness()) ? "signed" : "";
  return llvm::formatv("wire {0} [{1}:0]", signedModifier, width - 1);
}

// printRawDataFromAttr prints a string of the form <BIT_SIZE>'h<HEX_DATA>
// representing the dense element attribute.
void printRawDataFromAttr(DenseElementsAttr attr, raw_ostream &os) {
  auto iType = dyn_cast<IntegerType>(attr.getElementType());
  assert(iType);

  int32_t hexWidth = iType.getWidth() / 4;
  os << iType.getWidth() * attr.size() << "'h";
  auto attrIt = attr.value_end<APInt>();
  for (size_t i = 0; i < (size_t)attr.size(); ++i) {
    llvm::SmallString<40> s;
    (*--attrIt).toString(s, 16, false);
    os << std::string(hexWidth - s.str().size(), '0') << s;
  }
}

llvm::SmallString<128> variableLoadStr(
    MemRefType memRefType, ValueRange indices, unsigned int width,
    std::function<std::string(Value)> valueToString) {
  auto idx = flattenIndexExpression(
      memRefType, indices, [&](Value value) -> std::string {
        if (auto constOp =
                dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
          return std::to_string(cast<IntegerAttr>(constOp.getValue()).getInt());
        }
        return valueToString(value);
      });
  auto wrappedIdx = indices.size() == 1 ? idx : llvm::formatv("({0})", idx);
  return llvm::formatv("{0} + {2} * {1} : {2} * {1}", width - 1, wrappedIdx,
                       width);
}

struct CtlzValueStruct {
  std::string temp32;
  std::string temp16;
  std::string temp8;
  std::string temp4;
};

// ctlzStructForResult constructs a struct that holds the values needed to
// compute the count leading zeros operation on i32.
// TODO(b/288881554): Support arbitrary bit widths.
CtlzValueStruct ctlzStructForResult(StringRef result) {
  return CtlzValueStruct{.temp32 = llvm::formatv("{0}_{1}", result, "temp32"),
                         .temp16 = llvm::formatv("{0}_{1}", result, "temp16"),
                         .temp8 = llvm::formatv("{0}_{1}", result, "temp8"),
                         .temp4 = llvm::formatv("{0}_{1}", result, "temp4")};
}

}  // namespace

void registerToVerilogTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-verilog", "translate from arithmetic to verilog",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToVerilog(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<arith::ArithDialect, func::FuncDialect,
                        memref::MemRefDialect, affine::AffineDialect,
                        secret::SecretDialect, math::MathDialect>();
      });
}

LogicalResult translateToVerilog(Operation *op, llvm::raw_ostream &os,
                                 std::optional<llvm::StringRef> moduleName) {
  return translateToVerilog(op, os, moduleName, /*allowSecretOps=*/false);
}

LogicalResult translateToVerilog(Operation *op, llvm::raw_ostream &os,
                                 std::optional<llvm::StringRef> moduleName,
                                 bool allowSecretOps) {
  if (!allowSecretOps) {
    auto result = op->walk([&](Operation *op) -> WalkResult {
      if (isa<secret::SecretDialect>(op->getDialect())) {
        op->emitError("allowSecretOps is false, but encountered a secret op.");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) return failure();
  }

  VerilogEmitter emitter(os);
  LogicalResult result = emitter.translate(*op, moduleName);
  return result;
}

LogicalResult translateToVerilog(Operation *op, llvm::raw_ostream &os) {
  return translateToVerilog(op, os, std::nullopt);
}

VerilogEmitter::VerilogEmitter(raw_ostream &os) : os_(os), value_count_(0) {}

LogicalResult VerilogEmitter::translate(
    Operation &op, std::optional<llvm::StringRef> moduleName) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Ops that use moduleName
          .Case<ModuleOp, func::FuncOp, secret::GenericOp>(
              [&](auto op) { return printOperation(op, moduleName); })
          // Func ops.
          .Case<func::CallOp>([&](auto op) { return printOperation(op); })
          // Return-like ops
          .Case<func::ReturnOp, secret::YieldOp>(
              [&](auto op) { return printReturnLikeOp(op.getOperands()); })
          // Arithmetic ops.
          .Case<arith::ConstantOp>([&](arith::ConstantOp op) {
            if (auto iAttr = dyn_cast<IndexType>(op.getValue().getType())) {
              // We can skip translating declarations of index constants. If the
              // index is used in a subsequent load, e.g.
              //   %1 = arith.constant 1 : index
              //   %2 = arith.load %foo[%1] : memref<3xi8>
              // then the load's constant index value can be inferred directly
              // when translating the load operation, and we do not need to
              // declare the constant. For example, this would translate to
              //   v2 = vFoo[15:8];
              value_to_wire_name_.insert(std::make_pair(
                  op.getResult(),
                  std::to_string(cast<IntegerAttr>(op.getValue()).getInt())));
              return success();
            }
            return printOperation(op);
          })
          .Case<arith::AddIOp, arith::CmpIOp, arith::ExtSIOp, arith::ExtUIOp,
                arith::IndexCastOp, arith::MaxSIOp, arith::MinSIOp,
                arith::MulIOp, arith::SelectOp, arith::ShLIOp, arith::ShRSIOp,
                arith::ShRUIOp, arith::SubIOp, arith::TruncIOp, arith::AndIOp>(
              [&](auto op) { return printOperation(op); })
          // Custom math ops.
          .Case<math::CountLeadingZerosOp>(
              [&](auto op) { return printOperation(op); })
          // Memref ops.
          .Case<memref::DeallocOp>([&](auto op) { return success(); })
          .Case<memref::GlobalOp>([&](auto op) {
            // This is a no-op: Globals are not translated inherently, rather
            // their users get_globals are translated at the function level.
            return success();
          })
          .Case<memref::GetGlobalOp>([&](auto op) {
            // This is a no-op: GetGlobals are translated to a wire assignment
            // of their underlying constant global value during FuncOp
            // translation, when the MLIR module is known.
            return success();
          })
          .Case<memref::AllocOp>([&](auto op) {
            // This is a no-op. Memref allocations are translated to a wire
            // declaration during FuncOp translation.
            return success();
          })
          .Case<memref::LoadOp, memref::StoreOp>(
              [&](auto op) { return printOperation(op); })
          // Affine ops.
          .Case<affine::AffineParallelOp, affine::AffineLoadOp,
                affine::AffineStoreOp, affine::AffineYieldOp>(
              [&](auto op) { return printOperation(op); })
          .Case<UnrealizedConversionCastOp>(
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

LogicalResult VerilogEmitter::printOperation(
    ModuleOp moduleOp, std::optional<llvm::StringRef> moduleName) {
  // We have no use in separating things by modules, so just descend
  // to the underlying ops and continue.
  for (Operation &op : moduleOp) {
    if (failed(translate(op, moduleName))) {
      return failure();
    }
  }

  return success();
}

LogicalResult VerilogEmitter::printFunctionLikeOp(
    Operation *op, llvm::StringRef verilogModuleName,
    ArrayRef<BlockArgument> arguments, TypeRange resultTypes,
    Region::BlockListType::iterator blocksBegin,
    Region::BlockListType::iterator blocksEnd) {
  /*
   *  A func op translates as follows, noting the internal variable wires
   *  need to be defined at the beginning of the module.
   *
   *    module main(
   *      input wire [7:0] arg0,
   *      input wire [7:0] arg1,
   *      ... ,
   *      output wire [7:0] _out_1,
   *      output wire [7:0] _out_2,
   *      ...
   *    );
   *      wire [31:0] x0;
   *      wire [31:0] x1000;
   *      wire [31:0] x1001;
   *      ...
   *    endmodule
   */
  os_ << "module " << verilogModuleName << "(\n";
  os_.indent();
  llvm::SmallVector<std::string, 4> argsToPrint;
  for (auto arg : arguments) {
    std::string result;
    llvm::raw_string_ostream ss(result);
    ss << "input ";
    if (isa<IndexType>(arg.getType())) {
      if (failed(emitIndexType(arg, ss))) {
        arg.getParentBlock()->getParentOp()->emitError()
            << "failed to emit index value " << arg;
        return failure();
      }
    } else if (failed(emitType(arg.getType(), ss))) {
      arg.getParentBlock()->getParentOp()->emitError()
          << "failed to emit type " << arg.getType();
      return failure();
    }
    ss << " " << getOrCreateName(arg);
    argsToPrint.push_back(ss.str());
  }

  // Outputs must be declared as arguments as well,
  // in the same declaration list.
  for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
    std::string result;
    llvm::raw_string_ostream ss(result);
    ss << "output ";
    if (isa<IndexType>(resultType)) {
      op->emitError() << "cannot emit index type as output";
      return failure();
    }

    if (failed(emitType(resultType, ss))) {
      op->emitError() << "failed to emit output type " << resultType;
      return failure();
    }

    ss << " " << getOrCreateOutputWireName(index);
    argsToPrint.push_back(ss.str());
  }
  os_ << llvm::join(argsToPrint.begin(), argsToPrint.end(), ",\n");

  // End of module header
  os_.unindent();
  os_ << "\n);\n";

  // Module body
  os_.indent();

  // Wire declarations.
  // Look for any op outputs, which are interleaved throughout the function
  // body. Collect any globals used.
  llvm::SmallVector<memref::GetGlobalOp> getGlobals;
  WalkResult result =
      op->walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (auto globalOp = dyn_cast<memref::GetGlobalOp>(op)) {
          getGlobals.push_back(globalOp);
        }
        if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(op)) {
          // IndexCastOp's are a layer of indirection in the arithmetic
          // dialect that is unneeded in Verilog. A wire declaration is not
          // needed. Simply remove the indirection by adding a map from the
          // index-casted result value to the input integer value.
          auto retVal = indexCastOp.getResult();
          if (!value_to_wire_name_.contains(retVal)) {
            value_to_wire_name_.insert(std::make_pair(
                retVal, getOrCreateName(indexCastOp.getIn()).str()));
          }
          return WalkResult::advance();
        }
        if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
          if (auto indexType =
                  dyn_cast<IndexType>(constantOp.getResult().getType())) {
            // Skip index constants: Verilog can use the value inline.
            return WalkResult::advance();
          }
        }
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
          auto inputType = (*castOp.getInputs().begin()).getType();
          auto returnType = (*castOp.getResults().begin()).getType();
          if (!mlir::isa<IntegerType>(inputType) ||
              !mlir::isa<IntegerType>(returnType)) {
            return WalkResult(op->emitError(
                "unable to support unrealized conversion cast "
                "op, expected conversion between integer types."));
          }
        }
        for (OpResult result : op->getResults()) {
          if (failed(emitWireDeclaration(result))) {
            return WalkResult(op->emitError()
                              << "unable to declare result variable of type "
                              << result.getType());
          }
        }
        // Also generate intermediate result values the CTLZ computation.
        if (auto ctlzOp = dyn_cast<math::CountLeadingZerosOp>(op)) {
          auto *ctx = op->getContext();
          auto ctlzStruct =
              ctlzStructForResult(getOrCreateName(ctlzOp.getResult()));
          llvm::SmallVector<std::pair<StringRef, int>, 4> tempWires = {
              {ctlzStruct.temp32, 32},
              {ctlzStruct.temp16, 16},
              {ctlzStruct.temp8, 8},
              {ctlzStruct.temp4, 4}};
          for (auto tempWire : tempWires) {
            if (failed(emitType(IntegerType::get(ctx, tempWire.second)))) {
              return failure();
            }
            os_ << " " << tempWire.first << ";\n";
          }
        }
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) return failure();

  auto module = op->getParentOfType<ModuleOp>();
  assert(module);

  // Assign global values while we have access to the top-level module.
  if (!getGlobals.empty()) {
    for (memref::GetGlobalOp getGlobalOp : getGlobals) {
      auto global = cast<memref::GlobalOp>(
          module.lookupSymbol(getGlobalOp.getNameAttr()));
      auto cstAttr = mlir::dyn_cast_or_null<DenseElementsAttr>(
          global.getConstantInitValue());
      if (!cstAttr) {
        return failure();
      }

      os_ << "assign " << getOrCreateName(getGlobalOp.getResult()) << " = ";
      printRawDataFromAttr(cstAttr, os_);
      os_ << ";\n";
    }
  }

  os_ << "\n";
  while (blocksBegin != blocksEnd) {
    for (Operation &op : blocksBegin->getOperations()) {
      if (failed(translate(op, std::nullopt))) {
        return failure();
      }
    }
    blocksBegin++;
  }
  os_.unindent();
  os_ << "endmodule\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(
    func::FuncOp funcOp, std::optional<llvm::StringRef> moduleName) {
  auto *blocks = &funcOp.getBlocks();
  return printFunctionLikeOp(
      funcOp.getOperation(), moduleName.value_or(funcOp.getName()),
      funcOp.getArguments(), funcOp.getFunctionType().getResults(),
      blocks->begin(), blocks->end());
}

LogicalResult VerilogEmitter::printReturnLikeOp(ValueRange returnValues) {
  // Return is an assignment to the output wire
  // e.g., assign out = x1200;
  if (returnValues.empty()) {
    return success();
  }
  for (auto [index, result] : llvm::enumerate(returnValues)) {
    os_ << "assign " << getOutputWireName(index) << " = "
        << getName(returnValues[index]) << ";\n";
  }
  return success();
}

LogicalResult VerilogEmitter::printOperation(func::CallOp op) {
  // e.g., submodule submod_call(xInput0, xInput1, xOutput);
  std::string opName = (getOrCreateName(op.getResult(0)) + "_call").str();

  // Verilog only supports functions with a single return value.
  if (op.getResults().size() != 1) {
    return failure();
  }
  std::string funcArgs;
  for (auto arg : op.getArgOperands()) {
    funcArgs += getOrCreateName(arg).str() + ", ";
  }
  funcArgs += getOrCreateName(op.getResult(0));
  os_ << op.getCallee() << " " << opName << "(" << funcArgs << ");\n";
  return success();
}

LogicalResult VerilogEmitter::printBinaryOp(Value result, Value lhs, Value rhs,
                                            std::string_view op) {
  emitAssignPrefix(result);
  os_ << getName(lhs) << " " << op << " " << getName(rhs) << ";\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(arith::AddIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "+");
}

LogicalResult VerilogEmitter::printOperation(arith::AndIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "&");
}

LogicalResult VerilogEmitter::printOperation(arith::CmpIOp op) {
  switch (op.getPredicate()) {
    // For eq and ne, verilog has multiple operators. == and === are
    // equivalent, except for the special values X (unknown default initial
    // state) and Z (high impedance state), which are irrelevant for our
    // purposes. Ditto for
    // != and !==.
    case arith::CmpIPredicate::eq:
      return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "==");
    case arith::CmpIPredicate::ne:
      return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "!=");
    // For comparison ops, signedness is important, but the semnatics of the
    // verilog operation are, in our case, determined by whether the operands
    // have a `signed` modifier on their declarations. See `emitType`.
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "<");
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "<=");
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">");
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">=");
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

LogicalResult VerilogEmitter::printOperation(arith::ConstantOp op) {
  Attribute attr = op.getValue();

  APInt value;
  bool isSigned = true;
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      value = iAttr.getValue();
      isSigned = shouldMapToSigned(iType.getSignedness());
    } else if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      return success();
    }
  }

  SmallString<128> strValue;
  value.toString(strValue, 10, isSigned, false);

  emitAssignPrefix(op.getResult());
  os_ << strValue << ";\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(UnrealizedConversionCastOp op) {
  // e.g. assign x0 = $signed(v100);
  // or   assign x0 = $unsigned(v100);
  auto inputIt = op.getInputs().begin();
  IntegerType outputType = dyn_cast<IntegerType>(*op.getResultTypes().begin());
  bool isSigned = shouldMapToSigned(outputType.getSignedness());
  for (auto res : op.getResults()) {
    emitAssignPrefix(res);
    os_ << (isSigned ? "$" : "$un") << "signed(" << getOrCreateName(*inputIt++)
        << ");\n";
  }
  return success();
}

LogicalResult VerilogEmitter::printOperation(arith::ExtSIOp op) {
  // E.g., assign x0 = {{24{arg0[7]}}, arg0};
  Value input = op.getIn();
  auto srcType = dyn_cast<IntegerType>(input.getType());
  auto dstType = dyn_cast<IntegerType>(op.getOut().getType());

  int extensionAmount = dstType.getWidth() - srcType.getWidth();
  StringRef arg = getOrCreateName(input);

  emitAssignPrefix(op.getResult());
  os_ << "{{" << extensionAmount << "{" << arg << "[" << srcType.getWidth() - 1
      << "]}}, " << arg << "};\n";

  return success();
}

LogicalResult VerilogEmitter::printOperation(arith::ExtUIOp op) {
  // E.g., assign x0 = {{24{1'b0}, arg0};
  Value input = op.getIn();
  auto srcType = dyn_cast<IntegerType>(input.getType());
  auto dstType = dyn_cast<IntegerType>(op.getOut().getType());

  int extensionAmount = dstType.getWidth() - srcType.getWidth();
  StringRef arg = getOrCreateName(input);

  emitAssignPrefix(op.getResult());
  os_ << "{{" << extensionAmount << "{1'b0}}, " << arg << "};\n";

  return success();
}

LogicalResult VerilogEmitter::printOperation(arith::IndexCastOp op) {
  // Verilog does not require casting integers to index types before use in an
  // array access index.
  return success();
}

LogicalResult VerilogEmitter::printOperation(arith::MulIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "*");
}

LogicalResult VerilogEmitter::printOperation(arith::SelectOp op) {
  emitAssignPrefix(op.getResult());
  os_ << getName(op.getCondition()) << " ? " << getName(op.getTrueValue())
      << " : " << getName(op.getFalseValue()) << ";\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(arith::MaxSIOp op) {
  emitAssignPrefix(op.getResult());
  auto lhs = getName(op.getLhs());
  auto rhs = getName(op.getRhs());
  os_ << lhs << " > " << rhs << " ? " << lhs << " : " << rhs << ";\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::MinSIOp op) {
  emitAssignPrefix(op.getResult());
  auto lhs = getName(op.getLhs());
  auto rhs = getName(op.getRhs());
  os_ << lhs << " < " << rhs << " ? " << lhs << " : " << rhs << ";\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(arith::ShLIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "<<");
}

LogicalResult VerilogEmitter::printOperation(arith::ShRSIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">>>");
}

LogicalResult VerilogEmitter::printOperation(arith::ShRUIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">>");
}

LogicalResult VerilogEmitter::printOperation(arith::SubIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "-");
}

LogicalResult VerilogEmitter::printOperation(arith::TruncIOp op) {
  // E.g., assign x0 = arg[7:0];
  auto dstType = dyn_cast<IntegerType>(op.getOut().getType());
  emitAssignPrefix(op.getResult());
  os_ << getOrCreateName(op.getIn()) << "[" << dstType.getWidth() - 1
      << ":0];\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(affine::AffineLoadOp op) {
  // This extracts the indexed bits from the flattened memref.
  auto iType = dyn_cast<IntegerType>(op.getMemRefType().getElementType());
  if (!iType) {
    return failure();
  }

  auto width = iType.getWidth();
  auto memrefStr = getOrCreateName(op.getMemref());

  affine::MemRefAccess access(op);
  auto optionalAccessIndex =
      getFlattenedAccessIndex(access, op.getMemRefType());

  if (optionalAccessIndex) {
    // This is a constant index accessor.
    emitAssignPrefix(op.getResult());
    auto flattenedBitIndex = optionalAccessIndex.value() * width;
    os_ << memrefStr << "[" << flattenedBitIndex + width - 1 << " : "
        << flattenedBitIndex << "];\n";
  } else {
    emitAssignPrefix(op.getResult());
    OpBuilder builder(op->getContext());
    auto indices = affine::expandAffineMap(builder, op->getLoc(), op.getMap(),
                                           op.getIndices());
    if (!indices.has_value()) {
      return failure();
    }
    for (auto index : indices.value()) {
      // We could avoid this constraint by printing the tree of defining
      // operations of the indices built by the affine map expander. For now,
      // this likely suffices.
      if (!value_to_wire_name_.contains(index) &&
          !dyn_cast_or_null<arith::ConstantOp>(index.getDefiningOp())) {
        return failure();
      }
    }
    os_ << memrefStr << "["
        << variableLoadStr(
               op.getMemRefType(), indices.value(), width,
               [&](Value value) { return getOrCreateName(value).str(); })
        << "];\n";
  }

  return success();
}

LogicalResult VerilogEmitter::printOperation(affine::AffineParallelOp op) {
  // An Affine parallel op can be emitted as a Verilog generate for loop to
  // replicate the loop body logic. See
  // https://fpgatutorial.com/verilog-generate/#:~:text=our%20verilog%20designs.-,Generate%20For%20Loop%20in%20Verilog,-We%20can%20use.
  if (op.getIVs().size() != 1) {
    op->emitError("only single induction var supported.");
    return failure();
  }
  auto iv = op.getIVs().front();
  auto min = op.getLowerBoundsMap().getSingleConstantResult();
  auto max = op.getUpperBoundsMap().getSingleConstantResult();
  auto step = op.getSteps().front();

  auto ivName = getOrCreateName(iv);
  os_ << "genvar " << ivName << ";\ngenerate\n";
  os_ << llvm::formatv("for ({0} = {1}; {0} < {2}; {0} = {0} + {3}) begin\n",
                       ivName, min, max, step);
  os_.indent();

  // Declare the wires local to the for loop.
  WalkResult result = op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      if (failed(emitWireDeclaration(result))) {
        return WalkResult(op->emitError()
                          << "unable to declare result variable of type "
                          << result.getType());
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();

  for (auto &operation : op.getBody()->getOperations()) {
    if (failed(translate(operation, std::nullopt))) {
      operation.emitError("failed to translate operation.");
      return failure();
    }
  }

  os_.unindent();
  os_ << "end\nendgenerate\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(affine::AffineYieldOp op) {
  if (op->getNumResults() > 0) {
    return failure();
  }
  return success();
}

LogicalResult VerilogEmitter::printOperation(memref::LoadOp op) {
  // This extracts the indexed bits from the flattened memref.
  auto iType = dyn_cast<IntegerType>(op.getMemRefType().getElementType());
  if (!iType) {
    return failure();
  }

  emitAssignPrefix(op.getResult());

  os_ << getOrCreateName(op.getMemref()) << "["
      << variableLoadStr(
             op.getMemRefType(), op.getIndices(), iType.getWidth(),
             [&](Value value) { return getOrCreateName(value).str(); })
      << "];\n";

  return success();
}

LogicalResult VerilogEmitter::printOperation(memref::StoreOp op) {
  // This extracts the indexed bits from the flattened memref.
  auto iType = dyn_cast<IntegerType>(op.getMemRefType().getElementType());
  if (!iType) {
    return failure();
  }

  os_ << "assign " << getOrCreateName(op.getMemref()) << "["
      << variableLoadStr(
             op.getMemRefType(), op.getIndices(), iType.getWidth(),
             [&](Value value) { return getOrCreateName(value).str(); })
      << "] = " << getOrCreateName(op.getOperands()[0]) << ";\n";

  return success();
}

LogicalResult VerilogEmitter::printOperation(affine::AffineStoreOp op) {
  // This extracts the indexed bits from the flattened memref.
  auto iType = dyn_cast<IntegerType>(op.getMemRefType().getElementType());
  if (!iType) {
    return failure();
  }

  auto width = iType.getWidth();
  affine::MemRefAccess access(op);
  auto optionalAccessIndex =
      getFlattenedAccessIndex(access, op.getMemRefType());
  if (optionalAccessIndex) {
    // This is a constant index accessor.
    auto flattenedBitIndex = optionalAccessIndex.value() * width;
    os_ << "assign " << getOrCreateName(op.getMemref()) << "["
        << flattenedBitIndex + width - 1 << " : " << flattenedBitIndex
        << "] = " << getOrCreateName(op.getOperands()[0]) << ";\n";
  } else {
    OpBuilder builder(op->getContext());
    auto indices = affine::expandAffineMap(builder, op->getLoc(), op.getMap(),
                                           op.getIndices());
    if (!indices.has_value()) {
      return failure();
    }
    for (auto index : indices.value()) {
      // We could avoid this constraint by printing the tree of defining
      // operations of the indices built by the affine map expander. For now,
      // this likely suffices.
      if (!value_to_wire_name_.contains(index) &&
          !dyn_cast_or_null<arith::ConstantOp>(index.getDefiningOp())) {
        return failure();
      }
    }
    os_ << "assign " << getOrCreateName(op.getMemref()) << "["
        << variableLoadStr(
               op.getMemRefType(), indices.value(), width,
               [&](Value value) { return getOrCreateName(value).str(); })
        << "] = " << getOrCreateName(op.getOperands()[0]) << ";\n";
  }

  return success();
}

LogicalResult VerilogEmitter::printOperation(math::CountLeadingZerosOp op) {
  // This adds a custom Verilog implementation of a count leading zeros op.
  auto resultStr = getOrCreateName(op.getResult());
  auto ctlzStruct = ctlzStructForResult(resultStr);
  auto opStr = getOrCreateName(op.getOperand());
  // We assign each bit of the temp32 result depending on the number of
  // leading zeros.
  os_ << "assign " << ctlzStruct.temp32 << "[31:5] = 27'b0;\n";
  os_ << "assign " << ctlzStruct.temp32 << "[4] = (" << opStr
      << "[31:16] == 16'b0);\n";
  os_ << "assign " << ctlzStruct.temp16 << " = " << ctlzStruct.temp32 << "[4]"
      << " ? " << opStr << "[15:0] : " << opStr << "[31:16];\n";
  os_ << "assign " << ctlzStruct.temp32 << "[3] = (" << opStr
      << "[15:8] == 8'b0);\n";
  os_ << "assign " << ctlzStruct.temp8 << " = " << ctlzStruct.temp32 << "[3]"
      << " ? " << ctlzStruct.temp16 << "[7:0] : " << ctlzStruct.temp16
      << "[15:8];\n";
  os_ << "assign " << ctlzStruct.temp32 << "[2] = (" << opStr
      << "[7:4] == 4'b0);\n";
  os_ << "assign " << ctlzStruct.temp4 << " = " << ctlzStruct.temp32 << "[2]"
      << " ? " << ctlzStruct.temp8 << "[3:0] : " << ctlzStruct.temp8
      << "[7:4];\n";
  os_ << "assign " << ctlzStruct.temp32 << "[1] = (" << opStr
      << "[3:2] == 2'b0);\n";
  os_ << "assign " << ctlzStruct.temp32 << "[0] = " << ctlzStruct.temp32
      << "[1] ? ~" << opStr << "[1] : ~" << opStr << "[3];\n";
  // If the input was zero, we add one to the result.
  os_ << "assign " << resultStr << " = (" << opStr << " == 32'b0) ? "
      << ctlzStruct.temp32 << " + 1 : " << ctlzStruct.temp32 << ";\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(
    mlir::heir::secret::GenericOp op,
    std::optional<llvm::StringRef> moduleName) {
  // Translate all the functions called in the module.
  auto module = op->getParentOfType<ModuleOp>();
  SetVector<Operation *> funcs;
  op->walk([&](func::CallOp callOp) {
    auto func = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
    funcs.insert(func.getOperation());
  });

  for (auto func : funcs) {
    if (failed(translate(*func, std::nullopt))) {
      return op->emitError() << "failed to translate function.";
    }
  }

  llvm::StringRef name;
  if (moduleName.has_value()) {
    name = moduleName.value();
  } else {
    // I wanted something more unique here, but an op.getLoc() doesn't print
    // as a valid verilog identifier. Maybe with enough string massaging it
    // could.
    name = "generic_body";
  }
  llvm::SmallVector<Type, 4> resultTypes;
  for (auto ty : op.getResultTypes())
    resultTypes.push_back(cast<secret::SecretType>(ty).getValueType());
  auto *blocks = &op.getRegion().getBlocks();
  return printFunctionLikeOp(op.getOperation(), name,
                             op.getRegion().getBlocks().front().getArguments(),
                             resultTypes, blocks->begin(), blocks->end());
}

LogicalResult VerilogEmitter::emitType(Type type) {
  return emitType(type, os_);
}

LogicalResult VerilogEmitter::emitType(Type type, raw_ostream &os) {
  if (auto idxType =
          dyn_cast<IndexType>(type)) {  // emit index types as 32-bit integers
    int32_t width = 32;
    IntegerType intTy = IntegerType::get(idxType.getContext(), width);
    return (os << wireDeclaration(intTy, width)), success();
  }
  if (auto iType = dyn_cast<IntegerType>(type)) {
    int32_t width = iType.getWidth();
    return (os << wireDeclaration(iType, width)), success();
  }
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    auto elementType = memRefType.getElementType();
    if (auto iType = dyn_cast<IntegerType>(elementType)) {
      int32_t flattenedWidth = memRefType.getNumElements() * iType.getWidth();
      return (os << wireDeclaration(iType, flattenedWidth)), success();
    }
  }
  return failure();
}

// Emit a wire declaration for an index value whose width corresponds to the
// smallest width required to index into any memref used by the value.
LogicalResult VerilogEmitter::emitIndexType(Value indexValue, raw_ostream &os) {
  // Operations on index types are not supported in this emitter, so we just
  // need to check the immediate users and inspect the memrefs they contain.
  int32_t biggestMemrefSize = 0;
  for (auto *user : indexValue.getUsers()) {
    int32_t memrefSize =
        llvm::TypeSwitch<Operation *, int32_t>(user)
            .Case<affine::AffineLoadOp, affine::AffineStoreOp, memref::LoadOp,
                  memref::StoreOp>(
                [&](auto op) { return op.getMemRefType().getNumElements(); })
            .Default([&](Operation *) { return 0; });
    biggestMemrefSize = std::max(biggestMemrefSize, memrefSize);
  }

  assert(biggestMemrefSize >= 0 &&
         "unexpected index value unused by any memref ops");
  auto widthBigint = APInt(64, biggestMemrefSize);
  int32_t width = widthBigint.isPowerOf2() ? widthBigint.logBase2()
                                           : widthBigint.logBase2() + 1;
  os << wireDeclaration(IntegerType::get(indexValue.getContext(), width),
                        width);
  return success();
}

void VerilogEmitter::emitAssignPrefix(Value result) {
  os_ << "assign " << getOrCreateName(result) << " = ";
}

LogicalResult VerilogEmitter::emitWireDeclaration(OpResult result) {
  Type ty = result.getType();
  if (mlir::isa<secret::SecretType>(ty))
    ty = cast<secret::SecretType>(ty).getValueType();
  if (failed(emitType(ty))) {
    return failure();
  }
  os_ << " " << getOrCreateName(result) << ";\n";
  return success();
}

StringRef VerilogEmitter::getOrCreateOutputWireName(int resultIndex) {
  if (resultIndex < (int)output_wire_names_.size()) {
    return output_wire_names_[resultIndex];
  }
  output_wire_names_.push_back(
      llvm::formatv("{0}{1}", kOutputPrefix, output_wire_names_.size()).str());
  return output_wire_names_.back();
}

StringRef VerilogEmitter::getOutputWireName(int resultIndex) {
  if (resultIndex < (int)output_wire_names_.size()) {
    return output_wire_names_[resultIndex];
  }
  llvm_unreachable(
      llvm::formatv("output wire name not found for index {0}", resultIndex)
          .str()
          .c_str());
}

StringRef VerilogEmitter::getOrCreateName(Value value,
                                          std::string_view prefix) {
  if (!value_to_wire_name_.contains(value)) {
    value_to_wire_name_.insert(std::make_pair(
        value, llvm::formatv("{0}{1}", prefix, ++value_count_).str()));
  }
  return value_to_wire_name_.at(value);
}

StringRef VerilogEmitter::getOrCreateName(BlockArgument arg) {
  return getOrCreateName(arg, "arg");
}

StringRef VerilogEmitter::getOrCreateName(Value value) {
  return getOrCreateName(value, "v");
}

// This is safe to call after an initial walk is performed over a function
// body to find each op result and call getOrCreateName on it. If this
// function is called and fails due to a missing key assertion, the bug is
// almost certainly in this file: MLIR ensures the input to this pass is in
// SSA form, and this file is responsible for populating value_to_wire_name_
// in a block before processing any operations in that block.
StringRef VerilogEmitter::getName(Value value) {
  return value_to_wire_name_.at(value);
}

}  // namespace heir
}  // namespace mlir
