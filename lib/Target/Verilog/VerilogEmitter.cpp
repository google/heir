#include "include/Target/Verilog/VerilogEmitter.h"

#include "include/Conversion/MemrefToArith/Utils.h"
#include "llvm/include/llvm/Support/FormatVariadic.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/Math/IR/Math.h" // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h" // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h" // from @llvm-project
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
void printRawDataFromAttr(mlir::DenseElementsAttr attr, raw_ostream &os) {
  auto iType = dyn_cast<IntegerType>(attr.getElementType());
  assert(iType);

  int32_t hexWidth = iType.getWidth() / 4;
  os << iType.getWidth() * attr.size() << "'h";
  auto attrIt = attr.value_end<mlir::APInt>();
  for (uint64_t i = 0; i < attr.size(); ++i) {
    llvm::SmallString<40> s;
    (*--attrIt).toStringSigned(s, 16);
    os << std::string(hexWidth - s.str().size(), '0') << s;
  }
}

llvm::SmallString<128> variableLoadStr(StringRef memref, StringRef index,
                                       unsigned int width) {
  return llvm::formatv("{0}[{1} + {2} * {3} : {2} * {3}]", memref, width - 1,
                       width, index);
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
  mlir::TranslateFromMLIRRegistration reg(
      "emit-verilog", "translate from arithmetic to verilog",
      [](Operation *op, llvm::raw_ostream &output) {
        return translateToVerilog(op, output);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect,
                        mlir::affine::AffineDialect, mlir::math::MathDialect>();
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
          .Case<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>(
              [&](auto op) { return printOperation(op); })
          // Arithmetic ops.
          .Case<mlir::arith::ConstantOp>([&](auto op) {
            if (auto iAttr = dyn_cast<IndexType>(op.getValue().getType())) {
              // We can skip translating declarations of index constants. If the
              // index is used in a subsequent load, e.g.
              //   %1 = arith.constant 1 : index
              //   %2 = arith.load %foo[%1] : memref<3xi8>
              // then the load's constant index value can be inferred directly
              // when translating the load operation, and we do not need to
              // declare the constant. For example, this would translate to
              //   v2 = vFoo[15:8];
              return success();
            }
            return printOperation(op);
          })
          .Case<mlir::arith::AddIOp, mlir::arith::CmpIOp, mlir::arith::ExtSIOp,
                mlir::arith::ExtUIOp, mlir::arith::IndexCastOp,
                mlir::arith::MulIOp, mlir::arith::SelectOp, mlir::arith::ShLIOp,
                mlir::arith::ShRSIOp, mlir::arith::ShRUIOp, mlir::arith::SubIOp,
                mlir::arith::TruncIOp, mlir::arith::AndIOp>(
              [&](auto op) { return printOperation(op); })
          // Custom math ops.
          .Case<mlir::math::CountLeadingZerosOp>(
              [&](auto op) { return printOperation(op); })
          // Memref ops.
          .Case<mlir::memref::GlobalOp>([&](auto op) {
            // This is a no-op: Globals are not translated inherently, rather
            // their users get_globals are translated at the function level.
            return mlir::success();
          })
          .Case<mlir::memref::GetGlobalOp>([&](auto op) {
            // This is a no-op: GetGlobals are translated to a wire assignment
            // of their underlying constant global value during FuncOp
            // translation, when the MLIR module is known.
            return mlir::success();
          })
          .Case<mlir::memref::AllocOp>([&](auto op) {
            // This is a no-op. Memref allocations are translated to a wire
            // declaration during FuncOp translation.
            return mlir::success();
          })
          .Case<mlir::memref::LoadOp>(
              [&](auto op) { return printOperation(op); })
          // Affine ops.
          .Case<mlir::affine::AffineLoadOp>(
              [&](auto op) { return printOperation(op); })
          .Case<mlir::affine::AffineStoreOp>(
              [&](auto op) { return printOperation(op); })
          .Case<mlir::UnrealizedConversionCastOp>(
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
  // body. Collect any globals used.
  llvm::SmallVector<mlir::memref::GetGlobalOp> get_globals;
  WalkResult result =
      funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (auto globalOp = dyn_cast<mlir::memref::GetGlobalOp>(op)) {
          get_globals.push_back(globalOp);
        }
        if (auto indexCastOp = dyn_cast<mlir::arith::IndexCastOp>(op)) {
          // IndexCastOp's are a layer of indirection in the arithmetic dialect
          // that is unneeded in Verilog. A wire declaration is not needed.
          // Simply remove the indirection by adding a map from the index-casted
          // result value to the input integer value.
          auto retVal = indexCastOp.getResult();
          if (!value_to_wire_name_.contains(retVal)) {
            value_to_wire_name_.insert(std::make_pair(
                retVal, getOrCreateName(indexCastOp.getIn()).str()));
          }
          return WalkResult::advance();
        }
        if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
          if (auto indexType =
                  dyn_cast<IndexType>(constantOp.getResult().getType())) {
            // Skip index constants: Verilog can use the value inline.
            return WalkResult::advance();
          }
        }
        if (auto castOp = dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
          auto inputType = (*castOp.getInputs().begin()).getType();
          auto returnType = (*castOp.getResults().begin()).getType();
          if (!inputType.isa<mlir::IntegerType>() ||
              !returnType.isa<mlir::IntegerType>()) {
            return WalkResult(op->emitError(
                "unable to support unrealized conversion cast "
                "op, expected conversion between integer types."));
          }
        }
        for (OpResult result : op->getResults()) {
          if (failed(emitWireDeclaration(result))) {
            return WalkResult(
                op->emitError("unable to declare result variable for op"));
          }
        }
        // Also generate intermediate result values the CTLZ computation.
        if (auto ctlzOp = dyn_cast<mlir::math::CountLeadingZerosOp>(op)) {
          auto ctx = op->getContext();
          auto ctlzStruct =
              ctlzStructForResult(getOrCreateName(ctlzOp.getResult()));
          llvm::SmallVector<std::pair<StringRef, int>, 4> tempWires = {
              {ctlzStruct.temp32, 32},
              {ctlzStruct.temp16, 16},
              {ctlzStruct.temp8, 8},
              {ctlzStruct.temp4, 4}};
          for (auto tempWire : tempWires) {
            if (failed(emitType(op->getLoc(),
                                IntegerType::get(ctx, tempWire.second)))) {
              return failure();
            }
            os_ << " " << tempWire.first << ";\n";
          }
        }
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) return failure();

  auto module = funcOp->getParentOfType<mlir::ModuleOp>();
  assert(module);

  // Assign global values while we have access to the top-level module.
  if (!get_globals.empty()) {
    for (mlir::memref::GetGlobalOp getGlobalOp : get_globals) {
      auto global = mlir::cast<mlir::memref::GlobalOp>(
          module.lookupSymbol(getGlobalOp.getNameAttr()));
      auto cstAttr =
          global.getConstantInitValue().dyn_cast_or_null<DenseElementsAttr>();
      if (!cstAttr) {
        return mlir::failure();
      }

      os_ << "assign " << getOrCreateName(getGlobalOp.getResult()) << " = ";
      printRawDataFromAttr(cstAttr, os_);
      os_ << ";\n";
    }
  }

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
  os_ << "endmodule\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::func::ReturnOp op) {
  // Return is an assignment to the output wire
  // e.g., assign out = x1200;

  // Only support one return value.
  auto retval = op.getOperands()[0];
  os_ << "assign " << kOutputName << " = " << getName(retval) << ";\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::func::CallOp op) {
  // e.g., submodule submod_call(xInput0, xInput1, xOutput);
  auto opName = getOrCreateName(op.getResult(0)) + "_call";

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

LogicalResult VerilogEmitter::printOperation(mlir::arith::AddIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "+");
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::AndIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "&");
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
      return success();
    }
  }

  SmallString<128> strValue;
  value.toString(strValue, 10, isSigned, false);

  emitAssignPrefix(op.getResult());
  os_ << strValue << ";\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(
    mlir::UnrealizedConversionCastOp op) {
  // e.g. assign x0 = $signed(v100);
  // or   assign x0 = $unsigned(v100);
  auto inputIt = op.getInputs().begin();
  mlir::IntegerType outputType =
      dyn_cast<mlir::IntegerType>(*op.getResultTypes().begin());
  bool isSigned = shouldMapToSigned(outputType.getSignedness());
  for (auto res : op.getResults()) {
    emitAssignPrefix(res);
    os_ << (isSigned ? "$" : "$un") << "signed(" << getOrCreateName(*inputIt++)
        << ");\n";
  }
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
      << "]}}, " << arg << "};\n";

  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::ExtUIOp op) {
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

LogicalResult VerilogEmitter::printOperation(mlir::arith::IndexCastOp op) {
  // Verilog does not require casting integers to index types before use in an
  // array access index.
  return mlir::success();
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::MulIOp op) {
  return printBinaryOp(op.getResult(), op.getLhs(), op.getRhs(), "*");
}

LogicalResult VerilogEmitter::printOperation(mlir::arith::SelectOp op) {
  emitAssignPrefix(op.getResult());
  os_ << getName(op.getCondition()) << " ? " << getName(op.getTrueValue())
      << " : " << getName(op.getFalseValue()) << ";\n";
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
  os_ << getOrCreateName(op.getIn()) << "[" << dstType.getWidth() - 1
      << ":0];\n";
  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::affine::AffineLoadOp op) {
  // This extracts the indexed bits from the flattened memref.
  auto iType = dyn_cast<IntegerType>(op.getMemRefType().getElementType());
  if (!iType) {
    return failure();
  }

  auto width = iType.getWidth();
  affine::MemRefAccess access(op);
  auto optionalAccessIndex =
      getFlattenedAccessIndex(access, op.getMemRefType());

  auto memrefStr = getOrCreateName(op.getMemref());
  if (optionalAccessIndex) {
    // This is a constant index accessor.
    emitAssignPrefix(op.getResult());
    auto flattenedBitIndex = optionalAccessIndex.value() * width;
    os_ << memrefStr << "[" << flattenedBitIndex + width - 1 << " : "
        << flattenedBitIndex << "];\n";
  } else {
    if (op.getMemRefType().getRank() > 1) {
      // TODO(b/284323495): Handle multi-dim variable access.
      return failure();
    }
    emitAssignPrefix(op.getResult());
    os_ << variableLoadStr(memrefStr, getOrCreateName(op.getIndices()[0]),
                           width)
        << ";\n";
  }

  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::memref::LoadOp op) {
  // This extracts the indexed bits from the flattened memref.
  auto iType = dyn_cast<IntegerType>(op.getMemRefType().getElementType());
  if (!iType) {
    return failure();
  }

  auto memrefStr = getOrCreateName(op.getMemref());
  auto indexStr = getOrCreateName(op.getIndices()[0]);
  auto width = iType.getWidth();

  if (op.getMemRefType().getRank() > 1) {
    // TODO(b/284323495): Handle multi-dim variable access.
    return failure();
  }

  emitAssignPrefix(op.getResult());
  os_ << variableLoadStr(memrefStr, indexStr, width) << ";\n";

  return success();
}

LogicalResult VerilogEmitter::printOperation(mlir::affine::AffineStoreOp op) {
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
    if (op.getMemRefType().getRank() > 1) {
      // TODO(b/284323495): Handle multi-dim variable access.
      return failure();
    }
    os_ << "assign " << getOrCreateName(op.getMemref()) << "[" << width - 1
        << " + " << width << " * " << getOrCreateName(op.getIndices()[0])
        << " : " << width << " * " << getOrCreateName(op.getIndices()[0])
        << "] = " << getOrCreateName(op.getOperands()[0]) << ";\n";
  }

  return success();
}

LogicalResult VerilogEmitter::printOperation(
    mlir::math::CountLeadingZerosOp op) {
  // This adds a custom Verilog implementation of a count leading zeros op.
  auto resultStr = getOrCreateName(op.getResult());
  auto ctlzStruct = ctlzStructForResult(resultStr);
  auto opStr = getOrCreateName(op.getOperand());
  // We assign each bit of the temp32 result depending on the number of leading
  // zeros.
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

LogicalResult VerilogEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    int32_t width = iType.getWidth();
    return (os_ << wireDeclaration(iType, width)), success();
  } else if (auto memRefType = dyn_cast<mlir::MemRefType>(type)) {
    auto elementType = memRefType.getElementType();
    if (auto iType = dyn_cast<IntegerType>(elementType)) {
      int32_t flattenedWidth = memRefType.getNumElements() * iType.getWidth();
      return (os_ << wireDeclaration(iType, flattenedWidth)), success();
    }
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
