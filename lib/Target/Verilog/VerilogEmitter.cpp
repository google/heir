#include "include/Target/Verilog/VerilogEmitter.h"

#include "llvm/include/llvm/Support/FormatVariadic.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h" // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h" // from @llvm-project

namespace mlir {
namespace heir {

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
          // // Func ops.
          .Case<mlir::func::FuncOp>([&](auto op) { return printOperation(op); })
          // Arithmetic ops.
          .Case<mlir::arith::ConstantOp>(
              [&](auto op) { return printOperation(op); })
          .Default([&](Operation &) {
            assert(0 && "unimplemented");
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status)) return failure();
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
  // Since a verilog module has only one output type, and the names are scoped
  // to the module, we can safely hard code `_out_` as every functions output
  // write name.
  os_ << " _out_";

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
  os_ << "\n";
  // return
  os_.unindent();
  os_ << "endmodule\n";

  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::AddIOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::CmpIOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::ConstantOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::ExtSIOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::MulIOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::SelectOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::ShLIOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::ShRSIOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::SubIOp op) {
  assert(0 && "unimplemented");
  return failure();
}
LogicalResult VerilogEmitter::printOperation(mlir::arith::TruncIOp op) {
  assert(0 && "unimplemented");
  return failure();
}

LogicalResult VerilogEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    int32_t width = iType.getWidth();
    if (width == 1) {
      return (os_ << "wire"), success();
    }
    return (os_ << "wire [" << width - 1 << ":0]"), success();
  }
  assert(0 && "unimplemented");
  return failure();
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
  if (!value_to_wire_name_.count(value)) {
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

}  // namespace heir
}  // namespace mlir
