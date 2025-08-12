//===- CombOps.cpp - Implement the Comb operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements combinational ops.
//
//===----------------------------------------------------------------------===//

#include "lib/Dialect/Comb/IR/CombOps.h"

#include <cstddef>
#include <optional>

#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project

namespace mlir {
namespace heir {
namespace comb {

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

ICmpPredicate ICmpOp::getFlippedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
    case ICmpPredicate::eq:
      return ICmpPredicate::eq;
    case ICmpPredicate::ne:
      return ICmpPredicate::ne;
    case ICmpPredicate::slt:
      return ICmpPredicate::sgt;
    case ICmpPredicate::sle:
      return ICmpPredicate::sge;
    case ICmpPredicate::sgt:
      return ICmpPredicate::slt;
    case ICmpPredicate::sge:
      return ICmpPredicate::sle;
    case ICmpPredicate::ult:
      return ICmpPredicate::ugt;
    case ICmpPredicate::ule:
      return ICmpPredicate::uge;
    case ICmpPredicate::ugt:
      return ICmpPredicate::ult;
    case ICmpPredicate::uge:
      return ICmpPredicate::ule;
    case ICmpPredicate::ceq:
      return ICmpPredicate::ceq;
    case ICmpPredicate::cne:
      return ICmpPredicate::cne;
    case ICmpPredicate::weq:
      return ICmpPredicate::weq;
    case ICmpPredicate::wne:
      return ICmpPredicate::wne;
  }
  llvm_unreachable("unknown comparison predicate");
}

bool ICmpOp::isPredicateSigned(ICmpPredicate predicate) {
  switch (predicate) {
    case ICmpPredicate::ult:
    case ICmpPredicate::ugt:
    case ICmpPredicate::ule:
    case ICmpPredicate::uge:
    case ICmpPredicate::ne:
    case ICmpPredicate::eq:
    case ICmpPredicate::cne:
    case ICmpPredicate::ceq:
    case ICmpPredicate::wne:
    case ICmpPredicate::weq:
      return false;
    case ICmpPredicate::slt:
    case ICmpPredicate::sgt:
    case ICmpPredicate::sle:
    case ICmpPredicate::sge:
      return true;
  }
  llvm_unreachable("unknown comparison predicate");
}

/// Returns the predicate for a logically negated comparison, e.g. mapping
/// EQ => NE and SLE => SGT.
ICmpPredicate ICmpOp::getNegatedPredicate(ICmpPredicate predicate) {
  switch (predicate) {
    case ICmpPredicate::eq:
      return ICmpPredicate::ne;
    case ICmpPredicate::ne:
      return ICmpPredicate::eq;
    case ICmpPredicate::slt:
      return ICmpPredicate::sge;
    case ICmpPredicate::sle:
      return ICmpPredicate::sgt;
    case ICmpPredicate::sgt:
      return ICmpPredicate::sle;
    case ICmpPredicate::sge:
      return ICmpPredicate::slt;
    case ICmpPredicate::ult:
      return ICmpPredicate::uge;
    case ICmpPredicate::ule:
      return ICmpPredicate::ugt;
    case ICmpPredicate::ugt:
      return ICmpPredicate::ule;
    case ICmpPredicate::uge:
      return ICmpPredicate::ult;
    case ICmpPredicate::ceq:
      return ICmpPredicate::cne;
    case ICmpPredicate::cne:
      return ICmpPredicate::ceq;
    case ICmpPredicate::weq:
      return ICmpPredicate::wne;
    case ICmpPredicate::wne:
      return ICmpPredicate::weq;
  }
  llvm_unreachable("unknown comparison predicate");
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

LogicalResult ReplicateOp::verify() {
  // The source must be equal or smaller than the dest type, and an even
  // multiple of it.  Both are already known to be signless integers.
  auto srcWidth = mlir::cast<IntegerType>(getOperand().getType()).getWidth();
  auto dstWidth = mlir::cast<IntegerType>(getType()).getWidth();
  if (srcWidth == 0)
    return emitOpError("replicate does not take zero bit integer");

  if (srcWidth > dstWidth)
    return emitOpError("replicate cannot shrink bitwidth of operand"),
           failure();

  if (dstWidth % srcWidth)
    return emitOpError("replicate must produce integer multiple of operand"),
           failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

static LogicalResult verifyUTBinOp(Operation* op) {
  if (op->getOperands().empty())
    return op->emitOpError("requires 1 or more args");
  return success();
}

LogicalResult AddOp::verify() { return verifyUTBinOp(*this); }

LogicalResult MulOp::verify() { return verifyUTBinOp(*this); }

LogicalResult AndOp::verify() { return verifyUTBinOp(*this); }

LogicalResult OrOp::verify() { return verifyUTBinOp(*this); }

LogicalResult XorOp::verify() { return verifyUTBinOp(*this); }

LogicalResult XNorOp::verify() { return verifyUTBinOp(*this); }

LogicalResult NandOp::verify() { return verifyUTBinOp(*this); }

LogicalResult NorOp::verify() { return verifyUTBinOp(*this); }

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

static unsigned getTotalWidth(ValueRange inputs) {
  unsigned resultWidth = 0;
  for (auto input : inputs) {
    resultWidth += mlir::cast<IntegerType>(input.getType()).getWidth();
  }
  return resultWidth;
}

LogicalResult ConcatOp::verify() {
  unsigned tyWidth = mlir::cast<IntegerType>(getType()).getWidth();
  unsigned operandsTotalWidth = getTotalWidth(getInputs());
  if (tyWidth != operandsTotalWidth)
    return emitOpError(
               "ConcatOp requires operands total width to "
               "match type width. operands "
               "totalWidth is")
           << operandsTotalWidth << ", but concatOp type width is " << tyWidth;

  return success();
}

void ConcatOp::build(OpBuilder& builder, OperationState& result, Value hd,
                     ValueRange tl) {
  result.addOperands(ValueRange{hd});
  result.addOperands(tl);
  unsigned hdWidth = mlir::cast<IntegerType>(hd.getType()).getWidth();
  result.addTypes(builder.getIntegerType(getTotalWidth(tl) + hdWidth));
}

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type>& results) {
  unsigned resultWidth = getTotalWidth(operands);
  results.push_back(IntegerType::get(context, resultWidth));
  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  unsigned srcWidth = mlir::cast<IntegerType>(getInput().getType()).getWidth();
  unsigned dstWidth = mlir::cast<IntegerType>(getType()).getWidth();
  if (getLowBit() >= srcWidth || srcWidth - getLowBit() < dstWidth)
    return emitOpError("from bit too large for input"), failure();

  return success();
}

LogicalResult TruthTableOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs >= sizeof(size_t) * 8)
    return emitOpError("Truth tables support a maximum of ")
           << sizeof(size_t) * 8 - 1 << " inputs on your platform";

  auto table = getLookupTable();
  if (table.getValue().getBitWidth() != (1ull << numInputs))
    return emitOpError("Expected lookup table int of 2^n bits");
  return success();
}

std::optional<mlir::ValueRange> TruthTableOp::getLookupTableInputs() {
  return mlir::ValueRange{getInputs()};
}

}  // namespace comb
}  // namespace heir
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "lib/Dialect/Comb/IR/Comb.cpp.inc"
