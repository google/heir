#include "include/Target/Verilog/VerilogEmitter.h"

#include "llvm/include/llvm/Support/FormatVariadic.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h" // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h" // from @llvm-project

namespace mlir {
namespace heir {

namespace {

// Since a verilog module has only one output value, and the names are scoped
// to the module, we can safely hard code `_out_` as every function's output
// wire name.
static constexpr std::string_view kOutputName = "_out_";

bool shouldMapToSigned(IntegerType::SignednessSemantics val) {
  switch (val) {
    case IntegerType::Signless:
      return true;
    case IntegerType::Signed:
      return true;
    case IntegerType::Unsigned:
      return false;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

}  // namespace

void registerToVerilogTranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "emit-verilog", "translate from arithmetic to verilog",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToVerilog(op, output);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
      });
}

LogicalResult translateToVerilog(Operation *op, llvm::raw_ostream &os) {
  VerilogEmitter emitter(os);
  LogicalResult result = emitter.translate(*op);
  return result;
}

VerilogEmitter::VerilogEmitter(raw_ostream &os) : os_(os), value_count_(0) {}

LogicalResult VerilogEmitter::translate(Operation &op) {
  LogicalResult status =
      llvm::TypeSwitch<Operation &, LogicalResult>(op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(op); })
          // Func ops.
          .Case<mlir::func::FuncOp, mlir::func::ReturnOp>(
              [&](auto op) { return printOperation(op); })
          // Arithmetic ops.
          .Case<mlir::arith::ConstantOp, mlir::arith::ConstantOp,
                mlir::arith::AddIOp, mlir::arith::CmpIOp,
                mlir::arith::ConstantOp, mlir::arith::ExtSIOp,
                mlir::arith::MulIOp, mlir::arith::SelectOp, mlir::arith::ShLIOp,
                mlir::arith::ShRSIOp, mlir::arith::ShRUIOp, mlir::arith::SubIOp,
                mlir::arith::TruncIOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) {
    op.emitOpError(llvm::formatv("Failed to translate op {0}", op.getName()));
    return failure();
  }
  os_ << "\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::ModuleOp moduleOp) {
  // We have no use in separating things by modules, so just descend
  // to the underlying ops and continue.
  for (Operation &op : moduleOp) {
    if (failed(translate(op))) {
      return failure();
    }
  }

  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::func::FuncOp funcOp) {
  /*
   *  A func op translates as follows, noting the internal variable wires
   *  need to be defined at the beginning of the module.
   *
   *    module main(
   *      input wire [7:0] arg0,
   *      input wire [7:0] arg1,
   *      ... ,
   *      output wire [7:0] out
   *    );
   *      wire [31:0] x0;
   *      wire [31:0] x1000;
   *      wire [31:0] x1001;
   *      ...
   *    endmodule
   */
  os_ << "module " << funcOp.getName() << "(\n";
  os_.indent();
  for (auto arg : funcOp.getArguments()) {
    // e.g., `input wire [31:0] arg0,`
    os_ << "input ";
    if (failed(emitType(arg.getLoc(), arg.getType()))) {
      return failure();
    }
    os_ << " " << getOrCreateName(arg) << ",\n";
  }

  // output arg declaration
  auto result_types = funcOp.getFunctionType().getResults();
  if (result_types.size() != 1) {
    emitError(funcOp.getLoc(),
              "Only functions with a single return type are supported");
    return failure();
  }
  os_ << "output ";
  if (failed(emitType(funcOp.getLoc(), result_types.front()))) {
    return failure();
  }
  os_ << " " << kOutputName;

  // End of module header
  os_.unindent();
  os_ << "\n);\n";

  // Module body
  os_.indent();

  // Wire declarations.
  // Look for any op outputs, which are interleaved throughout the function
  // body.
  WalkResult result =
      funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        for (OpResult result : op->getResults()) {
          if (failed(emitWireDeclaration(result))) {
            return WalkResult(
                op->emitError("unable to declare result variable for op"));
          }
        }
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) return failure();

  os_ << "\n";
  // ops
  for (mlir::Block &block : funcOp.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      if (failed(translate(op))) {
        return failure();
      }
    }
  }
  os_.unindent();
  os_ << "endmodule";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::func::ReturnOp op) {
  // Return is an assignment to the output wire
  // e.g., assign out = x1200;

  // Only support one return value.
  auto retval = op.getOperands()[0];
  os_ << "assign " << kOutputName << " = " << getName(retval) << ";";
  return success();
}

LogicalResult VerilogEmitter::printBinaryOp(Value result, Value lhs, Value rhs,
                                            std::string_view op) {
  emitAssignPrefix(result);
  os_ << getName(lhs) << " " << op << " " << getName(rhs) << ";";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::AddIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "+");
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::CmpIOp op) {
  switch (op.getPredicate()) {
    // For eq and ne, verilog has multiple operators. == and === are equivalent,
    // except for the special values X (unknown default initial state) and Z
    // (high impedance state), which are irrelevant for our purposes. Ditto for
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

LogicalResult VerilogEmitter::printOperation(mlir::arith::ConstantOp op) {
  Attribute attr = op.getValue();

  APInt value;
  bool isSigned = true;
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      value = iAttr.getValue();
      isSigned = shouldMapToSigned(iType.getSignedness());
    } else if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      value = iAttr.getValue();
      isSigned = false;
    }
  }

  SmallString<128> strValue;
  value.toString(strValue, 10, isSigned, false);

  emitAssignPrefix(op.getResult());
  os_ << strValue << ";";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::ExtSIOp op) {
  // E.g., assign x0 = {{24{arg0[7]}}, arg0};
  Value input = op.getIn();
  auto srcType = dyn_cast<IntegerType>(input.getType());
  auto dstType = dyn_cast<IntegerType>(op.getOut().getType());

  int extensionAmount = dstType.getWidth() - srcType.getWidth();
  StringRef arg = getOrCreateName(input);

  emitAssignPrefix(op.getResult());
  os_ << "{{" << extensionAmount << "{" << arg << "[" << srcType.getWidth() - 1
      << "]}}, " << arg << "};";

  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::MulIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "*");
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::SelectOp op) {
  emitAssignPrefix(op.getResult());
  os_ << getName(op.getCondition()) << " ? " << getName(op.getTrueValue())
      << " : " << getName(op.getFalseValue()) << ";";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::ShLIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "<<");
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::ShRSIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">>>");
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::ShRUIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), ">>");
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::SubIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "-");
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::TruncIOp op) {
  // E.g., assign x0 = arg[7:0];
  auto dstType = dyn_cast<IntegerType>(op.getOut().getType());
  emitAssignPrefix(op.getResult());
  os_ << getOrCreateName(op.getIn()) << "[" << dstType.getWidth() - 1 << ":0];";
  return success();
}

LogicalResult VerilogEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    int32_t width = iType.getWidth();
    if (width == 1) {
      return (os_ << "wire"), success();
    }
    // By default, all verilog operations are unsigned. However, comparison
    // operators might be comparing two signed values. There are two ways to
    // tell verilog to do a signed comparison: one is to use the builtin
    // $signed() function around both operands when executing the comparison op,
    // otherwise both operands need to be defined as wires/registers with the
    // signed keyword. Thankfully, MLIR requires a signed cmpi to have both its
    // operands have the same type, so if we encounter a signed cmpi MLIR op, we
    // are guaranteed that both its wire declarations have this signed modifier.
    std::string_view signedModifier =
        shouldMapToSigned(iType.getSignedness()) ? "signed " : "";
    return (os_ << "wire " << signedModifier << "[" << width - 1 << ":0]"),
           success();
  }
  return failure();
}

void VerilogEmitter::emitAssignPrefix(Value result) {
  os_ << "assign " << getOrCreateName(result) << " = ";
}

LogicalResult VerilogEmitter::emitWireDeclaration(OpResult result) {
  if (failed(emitType(result.getLoc(), result.getType()))) {
    return failure();
  }
  os_ << " " << getOrCreateName(result) << ";\n";
  return success();
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

// This is safe to call after an initial walk is performed over a function body
// to find each op result and call getOrCreateName on it. If this function is
// called and fails due to a missing key assertion, the bug is almost certainly
// in this file: MLIR ensures the input to this pass is in SSA form, and this
// file is responsible for populating value_to_wire_name_ in a block before
// processing any operations in that block.
StringRef VerilogEmitter::getName(Value value) {
  return value_to_wire_name_.at(value);
}

}  // namespace heir
}  // namespace mlir
