#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/CheddarGPU/IR/CheddarGPUAttributes.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUDialect.h"
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUEnums.h"
#include "llvm/include/llvm/ADT/ArrayRef.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "lib/Dialect/CheddarGPU/IR/CheddarGPUOps.cpp.inc"

namespace mlir {
namespace heir {
namespace cheddar_gpu {

namespace {

LogicalResult verifyMemRefWordWidth(Operation* op, MemRefType type) {
  Type elemType = type.getElementType();
  if (!elemType.isSignlessInteger(32) && !elemType.isSignlessInteger(64)) {
    return op->emitOpError("element type must be i32 or i64, but got '")
           << elemType << "'";
  }
  return success();
}

LogicalResult verifyLimbRange(Operation* op,
                              std::optional<ArrayRef<int64_t>> range,
                              int64_t numLimbs, StringRef name) {
  if (!range) return success();
  if (range->size() != 2) {
    return op->emitOpError(name)
           << " must have exactly 2 elements [offset, count]";
  }
  int64_t offset = (*range)[0];
  int64_t count = (*range)[1];
  if (offset < 0 || count <= 0 || offset + count > numLimbs) {
    return op->emitOpError()
           << "limb range [" << offset << ", " << count
           << "] out of bounds for memref with " << numLimbs << " limbs";
  }

  if (auto mod = op->getParentOfType<ModuleOp>()) {
    if (auto params = mod->getAttrOfType<ParamsAttr>("cheddar_gpu.params")) {
      if (offset + count > static_cast<int64_t>(params.getNumLimbs())) {
        return op->emitOpError() << "limb range [" << offset << ", " << count
                                 << "] exceeds module params limb count ("
                                 << params.getNumLimbs() << ")";
      }
    }
  }
  return success();
}

LogicalResult verifySameShape(Operation* op, MemRefType a, MemRefType b,
                              StringRef nameA, StringRef nameB) {
  if (a.getShape() != b.getShape()) {
    return op->emitOpError("operand shape mismatch between ")
           << nameA << " and " << nameB;
  }
  if (a.getElementType() != b.getElementType()) {
    return op->emitOpError("element type mismatch between ")
           << nameA << " and " << nameB;
  }
  return success();
}

static LogicalResult verifyNumAux(Operation* op, std::optional<int64_t> numAux,
                                  int64_t numLimbs) {
  if (numAux) {
    if (*numAux < 0 || *numAux > numLimbs) {
      return op->emitOpError("num_aux must be in [0, ")
             << numLimbs << "], but got " << *numAux;
    }
  }
  return success();
}

LogicalResult verifyEvkDigits(Operation* op, ValueRange evkBx, ValueRange evkAx,
                              int64_t degree, Type elemType) {
  if (evkBx.empty()) {
    return op->emitOpError("evaluation key digits cannot be empty");
  }
  if (evkBx.size() != evkAx.size()) {
    return op->emitOpError(
               "evaluation key digit counts must match between evk_bx (")
           << evkBx.size() << ") and evk_ax (" << evkAx.size() << ")";
  }
  for (size_t i = 0; i < evkBx.size(); ++i) {
    auto bxType = ::llvm::cast<MemRefType>(evkBx[i].getType());
    auto axType = ::llvm::cast<MemRefType>(evkAx[i].getType());
    if (failed(verifyMemRefWordWidth(op, bxType)) ||
        failed(verifyMemRefWordWidth(op, axType))) {
      return failure();
    }
    if (bxType.getRank() != 2 || axType.getRank() != 2) {
      return op->emitOpError("evaluation key digits must be rank 2 memrefs");
    }
    if (bxType.getDimSize(1) != degree || axType.getDimSize(1) != degree) {
      return op->emitOpError(
                 "evaluation key digit degree mismatch with output degree (")
             << degree << ")";
    }
    if (bxType.getElementType() != elemType ||
        axType.getElementType() != elemType) {
      return op->emitOpError(
          "evaluation key digit element type mismatch with output");
    }
  }
  return success();
}

}  // namespace

// === CopyOp ===

LogicalResult CopyOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }

  if (outType.getRank() != 2 || inType.getRank() != 2) {
    return emitOpError("copy requires 2D memref operands");
  }

  if (outType.getDimSize(1) != inType.getDimSize(1)) {
    return emitOpError("degree mismatch between in and out memrefs");
  }

  int64_t inLimbs = inType.getDimSize(0);
  int64_t outLimbs = outType.getDimSize(0);

  if (failed(
          verifyLimbRange(*this, getInLimbRange(), inLimbs, "in_limb_range")) ||
      failed(verifyLimbRange(*this, getOutLimbRange(), outLimbs,
                             "out_limb_range"))) {
    return failure();
  }

  int64_t inCount = getInLimbRange() ? (*getInLimbRange())[1] : inLimbs;
  int64_t outCount = getOutLimbRange() ? (*getOutLimbRange())[1] : outLimbs;
  if (inCount != outCount) {
    return emitOpError("limb count mismatch between in and out");
  }

  return success();
}

// === AddOp ===

LogicalResult AddOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto lhsType = ::llvm::cast<MemRefType>(getLhs().getType());
  auto rhsType = ::llvm::cast<MemRefType>(getRhs().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, lhsType)) ||
      failed(verifyMemRefWordWidth(*this, rhsType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, lhsType, "out", "lhs")) ||
      failed(verifySameShape(*this, outType, rhsType, "out", "rhs"))) {
    return failure();
  }

  if (getLhsForm() != getRhsForm()) {
    return emitOpError("form mismatch on add: lhs form (")
           << stringifyForm(getLhsForm()) << ") and rhs form ("
           << stringifyForm(getRhsForm()) << ") must match";
  }
  if (getLhsForm() != getOutForm()) {
    return emitOpError("form mismatch on add: input form (")
           << stringifyForm(getLhsForm()) << ") and output form ("
           << stringifyForm(getOutForm()) << ") must match";
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getLhsLimbRange(), numLimbs,
                             "lhs_limb_range")) ||
      failed(verifyLimbRange(*this, getRhsLimbRange(), numLimbs,
                             "rhs_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === SubOp ===

LogicalResult SubOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto lhsType = ::llvm::cast<MemRefType>(getLhs().getType());
  auto rhsType = ::llvm::cast<MemRefType>(getRhs().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, lhsType)) ||
      failed(verifyMemRefWordWidth(*this, rhsType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, lhsType, "out", "lhs")) ||
      failed(verifySameShape(*this, outType, rhsType, "out", "rhs"))) {
    return failure();
  }

  if (getLhsForm() != getRhsForm()) {
    return emitOpError("form mismatch on sub: lhs form (")
           << stringifyForm(getLhsForm()) << ") and rhs form ("
           << stringifyForm(getRhsForm()) << ") must match";
  }
  if (getLhsForm() != getOutForm()) {
    return emitOpError("form mismatch on sub: input form (")
           << stringifyForm(getLhsForm()) << ") and output form ("
           << stringifyForm(getOutForm()) << ") must match";
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getLhsLimbRange(), numLimbs,
                             "lhs_limb_range")) ||
      failed(verifyLimbRange(*this, getRhsLimbRange(), numLimbs,
                             "rhs_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === NegOp ===

LogicalResult NegOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (getInForm() != getOutForm()) {
    return emitOpError("neg requires matching input and output forms");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === Add2Op ===

LogicalResult Add2Op::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  if (failed(verifyMemRefWordWidth(*this, outBxType))) return failure();

  if (getLhsForm() != getRhsForm()) {
    return emitOpError("form mismatch on add_2: lhs form (")
           << stringifyForm(getLhsForm()) << ") and rhs form ("
           << stringifyForm(getRhsForm()) << ") must match";
  }
  if (getLhsForm() != getOutForm()) {
    return emitOpError("form mismatch on add_2: input form (")
           << stringifyForm(getLhsForm()) << ") and output form ("
           << stringifyForm(getOutForm()) << ") must match";
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getLhsLimbRange(), numLimbs,
                             "lhs_limb_range")) ||
      failed(verifyLimbRange(*this, getRhsLimbRange(), numLimbs,
                             "rhs_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === Sub2Op ===

LogicalResult Sub2Op::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  if (failed(verifyMemRefWordWidth(*this, outBxType))) return failure();

  if (getLhsForm() != getRhsForm()) {
    return emitOpError("form mismatch on sub_2: lhs form (")
           << stringifyForm(getLhsForm()) << ") and rhs form ("
           << stringifyForm(getRhsForm()) << ") must match";
  }
  if (getLhsForm() != getOutForm()) {
    return emitOpError("form mismatch on sub_2: input form (")
           << stringifyForm(getLhsForm()) << ") and output form ("
           << stringifyForm(getOutForm()) << ") must match";
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getLhsLimbRange(), numLimbs,
                             "lhs_limb_range")) ||
      failed(verifyLimbRange(*this, getRhsLimbRange(), numLimbs,
                             "rhs_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === Neg2Op ===

LogicalResult Neg2Op::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  if (failed(verifyMemRefWordWidth(*this, outBxType))) return failure();

  if (getInForm() != getOutForm()) {
    return emitOpError("neg_2 requires matching input and output forms");
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === AddConstOp ===

LogicalResult AddConstOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  auto cstType = ::llvm::cast<MemRefType>(getCst().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType)) ||
      failed(verifyMemRefWordWidth(*this, cstType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (cstType.getRank() != 1) {
    return emitOpError(
               "constant buffer must be rank 1 (per-limb), but got rank ")
           << cstType.getRank();
  }
  if (cstType.getDimSize(0) != outType.getDimSize(0)) {
    return emitOpError("constant buffer size (")
           << cstType.getDimSize(0) << ") must match limb count ("
           << outType.getDimSize(0) << ")";
  }

  if (getInForm() != getOutForm()) {
    return emitOpError("add_const requires matching input and output forms");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return success();
}

// === SubConstOp ===

LogicalResult SubConstOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  auto cstType = ::llvm::cast<MemRefType>(getCst().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType)) ||
      failed(verifyMemRefWordWidth(*this, cstType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (cstType.getRank() != 1) {
    return emitOpError(
               "constant buffer must be rank 1 (per-limb), but got rank ")
           << cstType.getRank();
  }
  if (cstType.getDimSize(0) != outType.getDimSize(0)) {
    return emitOpError("constant buffer size (")
           << cstType.getDimSize(0) << ") must match limb count ("
           << outType.getDimSize(0) << ")";
  }

  if (getInForm() != getOutForm()) {
    return emitOpError("sub_const requires matching input and output forms");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return success();
}

// === SubOppositeConstOp ===

LogicalResult SubOppositeConstOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  auto cstType = ::llvm::cast<MemRefType>(getCst().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType)) ||
      failed(verifyMemRefWordWidth(*this, cstType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (cstType.getRank() != 1) {
    return emitOpError(
               "constant buffer must be rank 1 (per-limb), but got rank ")
           << cstType.getRank();
  }
  if (cstType.getDimSize(0) != outType.getDimSize(0)) {
    return emitOpError("constant buffer size (")
           << cstType.getDimSize(0) << ") must match limb count ("
           << outType.getDimSize(0) << ")";
  }

  if (getInForm() != getOutForm()) {
    return emitOpError(
        "sub_opposite_const requires matching input and output forms");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return success();
}

// === MulOp ===

LogicalResult MulOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto lhsType = ::llvm::cast<MemRefType>(getLhs().getType());
  auto rhsType = ::llvm::cast<MemRefType>(getRhs().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, lhsType)) ||
      failed(verifyMemRefWordWidth(*this, rhsType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, lhsType, "out", "lhs")) ||
      failed(verifySameShape(*this, outType, rhsType, "out", "rhs"))) {
    return failure();
  }

  if (getLhsForm() != Form::Eval || getRhsForm() != Form::Eval ||
      getOutForm() != Form::Eval) {
    return emitOpError("mul requires eval form for all operands");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getLhsLimbRange(), numLimbs,
                             "lhs_limb_range")) ||
      failed(verifyLimbRange(*this, getRhsLimbRange(), numLimbs,
                             "rhs_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === MulConstOp ===

LogicalResult MulConstOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  auto cstType = ::llvm::cast<MemRefType>(getCst().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType)) ||
      failed(verifyMemRefWordWidth(*this, cstType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (cstType.getRank() != 1) {
    return emitOpError(
               "constant buffer must be rank 1 (per-limb), but got rank ")
           << cstType.getRank();
  }
  if (cstType.getDimSize(0) != outType.getDimSize(0)) {
    return emitOpError("constant buffer size (")
           << cstType.getDimSize(0) << ") must match limb count ("
           << outType.getDimSize(0) << ")";
  }

  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("mul_const requires eval form for all operands");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === CPAccumOp ===

LogicalResult CPAccumOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  if (failed(verifyMemRefWordWidth(*this, outType))) return failure();

  if (getOutForm() != Form::Eval) {
    return emitOpError("cp_accum requires eval form for output");
  }

  if (getCtSrcs().size() != getCommonSrcs().size()) {
    return emitOpError("cp_accum ct_srcs and common_srcs count must match");
  }
  if (getCtSrcs().empty()) {
    return emitOpError("cp_accum requires at least one source pair");
  }

  int64_t numLimbs = outType.getDimSize(0);
  int64_t outCount = numLimbs;
  if (getOutLimbRange()) {
    outCount = (*getOutLimbRange())[1] - (*getOutLimbRange())[0];
  }

  for (auto ct : getCtSrcs()) {
    auto ctType = ::llvm::cast<MemRefType>(ct.getType());
    if (failed(verifyMemRefWordWidth(*this, ctType))) return failure();
    if (ctType.getRank() != 2) {
      return emitOpError("ct_src must be rank 2 memref");
    }
    if (ctType.getDimSize(1) != outType.getDimSize(1)) {
      return emitOpError("ct_src degree must match out degree");
    }
    if (ctType.getDimSize(0) < outCount) {
      return emitOpError("ct_src limb count must be at least out limb count");
    }
  }

  for (auto common : getCommonSrcs()) {
    auto commonType = ::llvm::cast<MemRefType>(common.getType());
    if (failed(verifyMemRefWordWidth(*this, commonType))) return failure();
    if (commonType.getRank() == 1) {
      if (commonType.getDimSize(0) < outCount) {
        return emitOpError(
            "1D common_src size must be at least out limb count");
      }
    } else if (commonType.getRank() == 2) {
      if (failed(verifySameShape(*this, outType, commonType, "out",
                                 "common_src"))) {
        return failure();
      }
    } else {
      return emitOpError("common_src must be rank 1 or rank 2 memref");
    }
  }

  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range"))) {
    return failure();
  }
  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === CPAccum2Op ===

LogicalResult CPAccum2Op::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType))) {
    return failure();
  }

  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax"))) {
    return failure();
  }

  if (getOutForm() != Form::Eval) {
    return emitOpError("cp_accum_2 requires eval form for output");
  }

  if (getCtBxSrcs().size() != getCtAxSrcs().size() ||
      getCtBxSrcs().size() != getCommonSrcs().size()) {
    return emitOpError(
        "cp_accum_2 ct_bx_srcs, ct_ax_srcs, and common_srcs count must match");
  }
  if (getCtBxSrcs().empty()) {
    return emitOpError("cp_accum_2 requires at least one source tuple");
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  int64_t outCount = numLimbs;
  if (getOutLimbRange()) {
    outCount = (*getOutLimbRange())[1] - (*getOutLimbRange())[0];
  }

  for (size_t k = 0; k < getCtBxSrcs().size(); ++k) {
    auto ctBx = getCtBxSrcs()[k];
    auto ctAx = getCtAxSrcs()[k];
    auto ctBxType = ::llvm::cast<MemRefType>(ctBx.getType());
    auto ctAxType = ::llvm::cast<MemRefType>(ctAx.getType());
    if (failed(verifyMemRefWordWidth(*this, ctBxType)) ||
        failed(verifyMemRefWordWidth(*this, ctAxType))) {
      return failure();
    }
    if (ctBxType.getRank() != 2 || ctAxType.getRank() != 2) {
      return emitOpError("ct_bx and ct_ax must be rank 2 memrefs");
    }
    if (ctBxType.getDimSize(1) != outBxType.getDimSize(1) ||
        ctAxType.getDimSize(1) != outAxType.getDimSize(1)) {
      return emitOpError("ct degree must match out degree");
    }
    if (ctBxType.getDimSize(0) < outCount ||
        ctAxType.getDimSize(0) < outCount) {
      return emitOpError("ct limb count must be at least out limb count");
    }
  }

  for (auto common : getCommonSrcs()) {
    auto commonType = ::llvm::cast<MemRefType>(common.getType());
    if (failed(verifyMemRefWordWidth(*this, commonType))) return failure();
    if (commonType.getRank() == 1) {
      if (commonType.getDimSize(0) < outCount) {
        return emitOpError(
            "1D common_src size must be at least out limb count");
      }
    } else if (commonType.getRank() == 2) {
      if (failed(verifySameShape(*this, outBxType, commonType, "out_bx",
                                 "common_src"))) {
        return failure();
      }
    } else {
      return emitOpError("common_src must be rank 1 or rank 2 memref");
    }
  }

  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range"))) {
    return failure();
  }
  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === CPAccumAddOp ===

LogicalResult CPAccumAddOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto accType = ::llvm::cast<MemRefType>(getAcc().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, accType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, accType, "out", "acc"))) {
    return failure();
  }

  if (getOutForm() != Form::Eval) {
    return emitOpError("cp_accum_add requires eval form for output");
  }
  if (getAccForm() && *getAccForm() != Form::Eval) {
    return emitOpError("cp_accum_add requires eval form for accumulator");
  }

  if (getCtSrcs().size() != getCommonSrcs().size()) {
    return emitOpError("cp_accum_add ct_srcs and common_srcs count must match");
  }
  if (getCtSrcs().empty()) {
    return emitOpError("cp_accum_add requires at least one source pair");
  }

  int64_t numLimbs = outType.getDimSize(0);
  int64_t outCount = numLimbs;
  if (getOutLimbRange()) {
    outCount = (*getOutLimbRange())[1] - (*getOutLimbRange())[0];
  }

  for (auto ct : getCtSrcs()) {
    auto ctType = ::llvm::cast<MemRefType>(ct.getType());
    if (failed(verifyMemRefWordWidth(*this, ctType))) return failure();
    if (ctType.getRank() != 2) {
      return emitOpError("ct_src must be rank 2 memref");
    }
    if (ctType.getDimSize(1) != outType.getDimSize(1)) {
      return emitOpError("ct_src degree must match out degree");
    }
    if (ctType.getDimSize(0) < outCount) {
      return emitOpError("ct_src limb count must be at least out limb count");
    }
  }

  for (auto common : getCommonSrcs()) {
    auto commonType = ::llvm::cast<MemRefType>(common.getType());
    if (failed(verifyMemRefWordWidth(*this, commonType))) return failure();
    if (commonType.getRank() == 1) {
      if (commonType.getDimSize(0) < outCount) {
        return emitOpError(
            "1D common_src size must be at least out limb count");
      }
    } else if (commonType.getRank() == 2) {
      if (failed(verifySameShape(*this, outType, commonType, "out",
                                 "common_src"))) {
        return failure();
      }
    } else {
      return emitOpError("common_src must be rank 1 or rank 2 memref");
    }
  }

  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range"))) {
    return failure();
  }
  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === CPAccumAdd2Op ===

LogicalResult CPAccumAdd2Op::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto accBxType = ::llvm::cast<MemRefType>(getAccBx().getType());
  auto accAxType = ::llvm::cast<MemRefType>(getAccAx().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, accBxType)) ||
      failed(verifyMemRefWordWidth(*this, accAxType))) {
    return failure();
  }

  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(
          verifySameShape(*this, outBxType, accBxType, "out_bx", "acc_bx")) ||
      failed(
          verifySameShape(*this, outAxType, accAxType, "out_ax", "acc_ax"))) {
    return failure();
  }

  if (getOutForm() != Form::Eval) {
    return emitOpError("cp_accum_add_2 requires eval form for output");
  }
  if (getAccForm() && *getAccForm() != Form::Eval) {
    return emitOpError("cp_accum_add_2 requires eval form for accumulator");
  }

  if (getCtBxSrcs().size() != getCtAxSrcs().size() ||
      getCtBxSrcs().size() != getCommonSrcs().size()) {
    return emitOpError(
        "cp_accum_add_2 ct_bx_srcs, ct_ax_srcs, and common_srcs count must "
        "match");
  }
  if (getCtBxSrcs().empty()) {
    return emitOpError("cp_accum_add_2 requires at least one source tuple");
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  int64_t outCount = numLimbs;
  if (getOutLimbRange()) {
    outCount = (*getOutLimbRange())[1] - (*getOutLimbRange())[0];
  }

  for (size_t k = 0; k < getCtBxSrcs().size(); ++k) {
    auto ctBx = getCtBxSrcs()[k];
    auto ctAx = getCtAxSrcs()[k];
    auto ctBxType = ::llvm::cast<MemRefType>(ctBx.getType());
    auto ctAxType = ::llvm::cast<MemRefType>(ctAx.getType());
    if (failed(verifyMemRefWordWidth(*this, ctBxType)) ||
        failed(verifyMemRefWordWidth(*this, ctAxType))) {
      return failure();
    }
    if (ctBxType.getRank() != 2 || ctAxType.getRank() != 2) {
      return emitOpError("ct_bx and ct_ax must be rank 2 memrefs");
    }
    if (ctBxType.getDimSize(1) != outBxType.getDimSize(1) ||
        ctAxType.getDimSize(1) != outAxType.getDimSize(1)) {
      return emitOpError("ct degree must match out degree");
    }
    if (ctBxType.getDimSize(0) < outCount ||
        ctAxType.getDimSize(0) < outCount) {
      return emitOpError("ct limb count must be at least out limb count");
    }
  }

  for (auto common : getCommonSrcs()) {
    auto commonType = ::llvm::cast<MemRefType>(common.getType());
    if (failed(verifyMemRefWordWidth(*this, commonType))) return failure();
    if (commonType.getRank() == 1) {
      if (commonType.getDimSize(0) < outCount) {
        return emitOpError(
            "1D common_src size must be at least out limb count");
      }
    } else if (commonType.getRank() == 2) {
      if (failed(verifySameShape(*this, outBxType, commonType, "out_bx",
                                 "common_src"))) {
        return failure();
      }
    } else {
      return emitOpError("common_src must be rank 1 or rank 2 memref");
    }
  }

  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range"))) {
    return failure();
  }
  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === TensorOp ===

LogicalResult TensorOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto outRxType = ::llvm::cast<MemRefType>(getOutRx().getType());
  auto src1BxType = ::llvm::cast<MemRefType>(getSrc1Bx().getType());
  auto src1AxType = ::llvm::cast<MemRefType>(getSrc1Ax().getType());
  auto src2BxType = ::llvm::cast<MemRefType>(getSrc2Bx().getType());
  auto src2AxType = ::llvm::cast<MemRefType>(getSrc2Ax().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType))) return failure();

  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(
          verifySameShape(*this, outBxType, outRxType, "out_bx", "out_rx")) ||
      failed(
          verifySameShape(*this, outBxType, src1BxType, "out_bx", "src1_bx")) ||
      failed(
          verifySameShape(*this, outBxType, src1AxType, "out_bx", "src1_ax")) ||
      failed(
          verifySameShape(*this, outBxType, src2BxType, "out_bx", "src2_bx")) ||
      failed(
          verifySameShape(*this, outBxType, src2AxType, "out_bx", "src2_ax"))) {
    return failure();
  }

  if (getLhsForm() != Form::Eval || getRhsForm() != Form::Eval ||
      getOutForm() != Form::Eval) {
    return emitOpError("tensor requires eval form for all operands");
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  return verifyLimbRange(*this, getOutLimbRange(), numLimbs, "out_limb_range");
}

// === TensorSquareOp ===

LogicalResult TensorSquareOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto outRxType = ::llvm::cast<MemRefType>(getOutRx().getType());
  auto srcBxType = ::llvm::cast<MemRefType>(getSrcBx().getType());
  auto srcAxType = ::llvm::cast<MemRefType>(getSrcAx().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType))) return failure();

  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(
          verifySameShape(*this, outBxType, outRxType, "out_bx", "out_rx")) ||
      failed(
          verifySameShape(*this, outBxType, srcBxType, "out_bx", "src_bx")) ||
      failed(
          verifySameShape(*this, outBxType, srcAxType, "out_bx", "src_ax"))) {
    return failure();
  }

  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("tensor_square requires eval form for all operands");
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  return verifyLimbRange(*this, getOutLimbRange(), numLimbs, "out_limb_range");
}

// === MulImaginaryUnitOp ===

LogicalResult MulImaginaryUnitOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  auto iUnitType = ::llvm::cast<MemRefType>(getIUnit().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType)) ||
      failed(verifyMemRefWordWidth(*this, iUnitType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (iUnitType.getRank() != 1) {
    return emitOpError("i_unit buffer must be rank 1 (per-limb), but got rank ")
           << iUnitType.getRank();
  }
  if (iUnitType.getDimSize(0) != outType.getDimSize(0)) {
    return emitOpError("i_unit buffer size (")
           << iUnitType.getDimSize(0) << ") must match limb count ("
           << outType.getDimSize(0) << ")";
  }

  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("mul_imag_unit requires eval form for all operands");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return success();
}

// === PermuteOp ===

LogicalResult PermuteOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("permute requires eval form for all operands");
  }

  bool hasGalois = getGalois().has_value();
  bool hasConj = getConj();
  if (!hasGalois && !hasConj) {
    return emitOpError("permute requires either galois or conj attribute");
  }
  if (hasGalois && hasConj) {
    return emitOpError("cannot specify both galois and conj attributes");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === Permute2Op ===

LogicalResult Permute2Op::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  if (failed(verifyMemRefWordWidth(*this, outBxType))) return failure();

  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("permute_2 requires eval form for all operands");
  }

  bool hasGalois = getGalois().has_value();
  bool hasConj = getConj();
  if (!hasGalois && !hasConj) {
    return emitOpError("permute_2 requires either galois or conj attribute");
  }
  if (hasGalois && hasConj) {
    return emitOpError("cannot specify both galois and conj attributes");
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === PermuteAccumOp ===

LogicalResult PermuteAccumOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  if (failed(verifyMemRefWordWidth(*this, outType))) return failure();

  if (getOutForm() != Form::Eval) {
    return emitOpError("permute_accum requires eval form for output");
  }

  if (getAcc()) {
    auto accType = ::llvm::cast<MemRefType>(getAcc().getType());
    if (failed(verifySameShape(*this, outType, accType, "out", "acc"))) {
      return failure();
    }
  }

  if (getSrcs().size() != getGaloisFactors().size()) {
    return emitOpError(
        "permute_accum srcs count must match galois_factors count");
  }
  if (getSrcs().empty()) {
    return emitOpError("permute_accum requires at least one source");
  }

  for (auto src : getSrcs()) {
    auto srcType = ::llvm::cast<MemRefType>(src.getType());
    if (failed(verifySameShape(*this, outType, srcType, "out", "src"))) {
      return failure();
    }
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range"))) {
    return failure();
  }
  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === PermuteAccum2Op ===

LogicalResult PermuteAccum2Op::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax"))) {
    return failure();
  }

  if (getOutForm() != Form::Eval) {
    return emitOpError("permute_accum_2 requires eval form for output");
  }

  if (static_cast<bool>(getAccBx()) != static_cast<bool>(getAccAx())) {
    return emitOpError(
        "permute_accum_2 requires both acc_bx and acc_ax or neither");
  }
  if (getAccBx()) {
    auto accBxType = ::llvm::cast<MemRefType>(getAccBx().getType());
    auto accAxType = ::llvm::cast<MemRefType>(getAccAx().getType());
    if (failed(
            verifySameShape(*this, outBxType, accBxType, "out_bx", "acc_bx")) ||
        failed(
            verifySameShape(*this, outAxType, accAxType, "out_ax", "acc_ax"))) {
      return failure();
    }
  }

  if (getSrcsBx().size() != getSrcsAx().size()) {
    return emitOpError("permute_accum_2 srcs_bx and srcs_ax count must match");
  }
  if (getSrcsBx().size() != getGaloisFactors().size()) {
    return emitOpError(
        "permute_accum_2 srcs count must match galois_factors count");
  }
  if (getSrcsBx().empty()) {
    return emitOpError("permute_accum_2 requires at least one source tuple");
  }

  for (size_t k = 0; k < getSrcsBx().size(); ++k) {
    auto sBx = ::llvm::cast<MemRefType>(getSrcsBx()[k].getType());
    auto sAx = ::llvm::cast<MemRefType>(getSrcsAx()[k].getType());
    if (failed(verifySameShape(*this, outBxType, sBx, "out_bx", "src_bx")) ||
        failed(verifySameShape(*this, outAxType, sAx, "out_ax", "src_ax"))) {
      return failure();
    }
  }

  int64_t numLimbs = outBxType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range"))) {
    return failure();
  }
  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === NTTOp ===

LogicalResult NTTOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (getInForm() != Form::Coeff) {
    return emitOpError("ntt requires coeff form for input");
  }
  if (getOutForm() != Form::Eval) {
    return emitOpError("ntt requires eval form for output");
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }
  if (getOutLimbRange() && getInLimbRange() &&
      (*getOutLimbRange())[1] != (*getInLimbRange())[1]) {
    return emitOpError(
        "limb count mismatch between in_limb_range and out_limb_range");
  }

  if (getPrimeRange()) {
    if (getPrimeRange()->size() != 2) {
      return emitOpError(
          "prime_range must have exactly 2 elements [start, count]");
    }
    int64_t count = (*getPrimeRange())[1];
    int64_t expectedCount =
        getInLimbRange() ? (*getInLimbRange())[1] : numLimbs;
    if (count != expectedCount) {
      return emitOpError("prime_range count mismatch with limb range count");
    }
  }

  if (getSkipRange()) {
    if (getSkipRange()->size() != 2) {
      return emitOpError(
          "skip_range must have exactly 2 elements [skip_start, skip_end]");
    }
    int64_t skipStart = (*getSkipRange())[0];
    int64_t skipEnd = (*getSkipRange())[1];
    if (skipStart < 0 || skipEnd < skipStart || skipEnd > numLimbs) {
      return emitOpError("skip_range [")
             << skipStart << ", " << skipEnd
             << "] out of bounds for memref with " << numLimbs << " limbs";
    }
  }

  if (getPrimeIndices()) {
    int64_t expectedCount =
        getInLimbRange() ? (*getInLimbRange())[1] : numLimbs;
    if (getPrimeIndices()->size() != static_cast<size_t>(expectedCount)) {
      return emitOpError("prime_indices count mismatch with limb range count");
    }
  }

  if (getSkipCopySrc()) {
    if (!getSkipRange()) {
      return emitOpError("skip_copy_src requires skip_range to be set");
    }
    auto skipSrcType = ::llvm::cast<MemRefType>(getSkipCopySrc().getType());
    if (failed(verifyMemRefWordWidth(*this, skipSrcType))) {
      return failure();
    }
    if (skipSrcType.getElementType() != outType.getElementType()) {
      return emitOpError(
          "skip_copy_src element type mismatch with destination buffer");
    }
    if (skipSrcType.getRank() != 2 ||
        skipSrcType.getDimSize(1) != outType.getDimSize(1)) {
      return emitOpError(
          "skip_copy_src degree mismatch with destination buffer");
    }
    int64_t skipStart = (*getSkipRange())[0];
    int64_t skipEnd = (*getSkipRange())[1];
    int64_t skipCount = skipEnd - skipStart;
    int64_t skipInOffset = getSkipCopyInOffset();
    if (skipInOffset < 0) {
      return emitOpError("skip_copy_in_offset must be non-negative");
    }
    if (skipSrcType.getDimSize(0) < skipCount + skipInOffset) {
      return emitOpError("skip_copy_src buffer too small for skipped range");
    }
  }

  return success();
}

// === INTTOp ===

LogicalResult INTTOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }

  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }

  if (getInForm() != Form::Eval) {
    return emitOpError("intt requires eval form for input");
  }

  if (getEpilogue() == INTTEpilogue::MultConstNormalize) {
    if (getOutForm() != Form::CoeffCentered) {
      return emitOpError(
          "intt with mult_const_normalize epilogue requires coeff_centered "
          "output form");
    }
  } else {
    if (getOutForm() != Form::Coeff) {
      return emitOpError("intt with ")
             << stringifyINTTEpilogue(getEpilogue())
             << " epilogue requires coeff output form";
    }
  }

  if (getEpilogue() == INTTEpilogue::None) {
    if (getCst()) {
      return emitOpError(
          "intt with none epilogue does not accept constant buffer");
    }
  } else {
    if (!getCst()) {
      return emitOpError("intt with ") << stringifyINTTEpilogue(getEpilogue())
                                       << " epilogue requires constant buffer";
    }
  }

  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), numLimbs,
                             "in_limb_range"))) {
    return failure();
  }
  if (getOutLimbRange() && getInLimbRange() &&
      (*getOutLimbRange())[1] != (*getInLimbRange())[1]) {
    return emitOpError(
        "limb count mismatch between in_limb_range and out_limb_range");
  }

  int64_t limbCount =
      getOutLimbRange()
          ? (*getOutLimbRange())[1]
          : (getInLimbRange() ? (*getInLimbRange())[1] : numLimbs);
  int64_t inOffset = getInLimbRange() ? (*getInLimbRange())[0] : 0;

  if (getPrimeRange()) {
    if (getPrimeRange()->size() != 2) {
      return emitOpError(
          "prime_range must have exactly 2 elements [start, count]");
    }
    int64_t count = (*getPrimeRange())[1];
    if (count != limbCount) {
      return emitOpError("prime_range count mismatch with limb range count");
    }
  }

  if (getPrimeIndices()) {
    if (getPrimeIndices()->size() != static_cast<size_t>(limbCount)) {
      return emitOpError("prime_indices count mismatch with limb range count");
    }
  }

  if (getInSplit()) {
    if (getInSplit()->size() != 2) {
      return emitOpError(
          "in_split must have exactly 2 elements [split_count, extra_limbs]");
    }
    int64_t splitCount = (*getInSplit())[0];
    int64_t extraLimbs = (*getInSplit())[1];
    if (splitCount < 0 || splitCount > limbCount) {
      return emitOpError("in_split split_count must be in [0, limb_count]");
    }
    if (extraLimbs < 0) {
      return emitOpError("in_split extra_limbs must be non-negative");
    }
    int64_t requiredExtent = inOffset + limbCount;
    if (splitCount < limbCount) {
      requiredExtent += extraLimbs;
    }
    if (inType.getDimSize(0) < requiredExtent) {
      return emitOpError("input buffer too small for in_split extent");
    }
  }

  if (getCst()) {
    auto cstType = ::llvm::cast<MemRefType>(getCst().getType());
    if (failed(verifyMemRefWordWidth(*this, cstType))) return failure();
    if (cstType.getRank() != 1) {
      return emitOpError(
                 "constant buffer must be rank 1 (per-limb), but got rank ")
             << cstType.getRank();
    }
    if (cstType.getElementType() != outType.getElementType()) {
      return emitOpError(
          "constant buffer element type mismatch between out and cst");
    }
    int64_t expectedLimbs = numLimbs;
    if (getOutLimbRange()) {
      expectedLimbs = (*getOutLimbRange())[1];
    } else if (getInLimbRange()) {
      expectedLimbs = (*getInLimbRange())[1];
    }
    if (cstType.getDimSize(0) != expectedLimbs) {
      return emitOpError("constant buffer size (")
             << cstType.getDimSize(0) << ") must match limb count ("
             << expectedLimbs << ")";
    }
  }

  return success();
}

// === BConvOp ===

LogicalResult BConvOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  auto tableType = ::llvm::cast<MemRefType>(getTable().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType)) ||
      failed(verifyMemRefWordWidth(*this, tableType))) {
    return failure();
  }

  if (outType.getElementType() != inType.getElementType() ||
      outType.getElementType() != tableType.getElementType()) {
    return emitOpError("element type mismatch between out, in, and table");
  }

  if (getInForm() != Form::CoeffCentered) {
    return emitOpError("bconv requires coeff_centered input form");
  }
  if (getOutForm() != Form::Coeff) {
    return emitOpError("bconv requires coeff output form");
  }

  if (outType.getRank() != 2) {
    return emitOpError("out must have rank 2, but got rank ")
           << outType.getRank();
  }
  if (inType.getRank() != 2) {
    return emitOpError("in must have rank 2, but got rank ")
           << inType.getRank();
  }
  if (tableType.getRank() != 2) {
    return emitOpError("table must have rank 2, but got rank ")
           << tableType.getRank();
  }

  if (outType.getDimSize(1) != inType.getDimSize(1)) {
    return emitOpError("polynomial degree mismatch between out (")
           << outType.getDimSize(1) << ") and in (" << inType.getDimSize(1)
           << ")";
  }

  if (auto mod = (*this)->getParentOfType<ModuleOp>()) {
    if (auto params = mod->getAttrOfType<ParamsAttr>("cheddar_gpu.params")) {
      int64_t expectedDegree = 1LL << params.getLogN();
      if (outType.getDimSize(1) != expectedDegree) {
        return emitOpError("polynomial degree (")
               << outType.getDimSize(1)
               << ") does not match module params degree (" << expectedDegree
               << ")";
      }
    }
  }

  int64_t totalOutLimbs = outType.getDimSize(0);
  int64_t totalInLimbs = inType.getDimSize(0);

  if (failed(verifyLimbRange(*this, getOutLimbRange(), totalOutLimbs,
                             "out_limb_range")) ||
      failed(verifyLimbRange(*this, getInLimbRange(), totalInLimbs,
                             "in_limb_range"))) {
    return failure();
  }

  int64_t activeInLimbs =
      getInLimbRange() ? (*getInLimbRange())[1] : totalInLimbs;
  int64_t activeOutLimbs =
      getOutLimbRange() ? (*getOutLimbRange())[1] : totalOutLimbs;

  int64_t tableDstLen = tableType.getDimSize(0);
  int64_t tableSrcLen = tableType.getDimSize(1);

  if (tableSrcLen != activeInLimbs) {
    return emitOpError("table source length (")
           << tableSrcLen << ") must match input limb count (" << activeInLimbs
           << ")";
  }

  if (getSkipRange()) {
    if (getSkipRange()->size() != 2) {
      return emitOpError(
          "skip_range must have exactly 2 elements [skip_start, skip_end]");
    }
    int64_t skipStart = (*getSkipRange())[0];
    int64_t skipEnd = (*getSkipRange())[1];
    if (skipStart < 0 || skipEnd < skipStart || skipEnd > activeOutLimbs) {
      return emitOpError("skip_range [")
             << skipStart << ", " << skipEnd
             << "] out of bounds for active destination with " << activeOutLimbs
             << " limbs";
    }
    int64_t skippedCount = skipEnd - skipStart;
    if (tableDstLen != activeOutLimbs - skippedCount) {
      return emitOpError("table destination length (")
             << tableDstLen
             << ") must match active destination limbs minus skipped limbs ("
             << (activeOutLimbs - skippedCount) << ")";
    }
  } else {
    if (tableDstLen != activeOutLimbs) {
      return emitOpError("table destination length (")
             << tableDstLen << ") must match active destination limbs ("
             << activeOutLimbs << ")";
    }
  }

  if (getDstPrimes()) {
    if (getDstPrimes()->size() != static_cast<size_t>(tableDstLen)) {
      return emitOpError(
          "dst_primes length mismatch with table destination length");
    }
  }

  return success();
}

// === PseudoModUpOp ===

LogicalResult PseudoModUpOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }
  if (failed(verifySameShape(*this, outType, inType, "out", "in"))) {
    return failure();
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("pseudo_mod_up requires eval form for input and output");
  }
  return success();
}

// === ModUpOp ===

LogicalResult ModUpOp::verify() {
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  if (failed(verifyMemRefWordWidth(*this, inType))) return failure();
  if (getOuts().empty()) {
    return emitOpError("mod_up requires at least one output buffer");
  }
  auto firstOutType = ::llvm::cast<MemRefType>(getOuts().front().getType());
  if (failed(verifyMemRefWordWidth(*this, firstOutType))) return failure();

  for (auto outVal : getOuts()) {
    auto outType = ::llvm::cast<MemRefType>(outVal.getType());
    if (failed(verifySameShape(*this, firstOutType, outType, "outs[0]",
                               "outs[i]"))) {
      return failure();
    }
  }

  if (firstOutType.getDimSize(1) != inType.getDimSize(1)) {
    return emitOpError("polynomial degree mismatch between output (")
           << firstOutType.getDimSize(1) << ") and input ("
           << inType.getDimSize(1) << ")";
  }
  if (static_cast<int64_t>(getLevel()) < 0 &&
      (getLevel() != -1 ||
       firstOutType.getDimSize(0) != inType.getDimSize(0) + 1)) {
    return emitOpError("level must be non-negative");
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("mod_up requires eval form for input and output");
  }
  return success();
}

// === ModDownOp ===

LogicalResult ModDownOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }
  if (outType.getDimSize(1) != inType.getDimSize(1)) {
    return emitOpError("polynomial degree mismatch between output (")
           << outType.getDimSize(1) << ") and input (" << inType.getDimSize(1)
           << ")";
  }
  if (outType.getDimSize(0) >= inType.getDimSize(0)) {
    return emitOpError("output limb count (")
           << outType.getDimSize(0)
           << ") must be strictly less than input limb count ("
           << inType.getDimSize(0) << ")";
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("mod_down requires eval form for input and output");
  }
  return success();
}

// === RescaleOp ===

LogicalResult RescaleOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }
  if (outType.getDimSize(1) != inType.getDimSize(1)) {
    return emitOpError("polynomial degree mismatch between output (")
           << outType.getDimSize(1) << ") and input (" << inType.getDimSize(1)
           << ")";
  }
  if (getLevel() <= 0) {
    return emitOpError("rescale is only valid for level > 0, got level ")
           << getLevel();
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("rescale requires eval form for input and output");
  }
  return success();
}

// === ModDownRescaleOp ===

LogicalResult ModDownRescaleOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }
  if (outType.getDimSize(1) != inType.getDimSize(1)) {
    return emitOpError("polynomial degree mismatch between output (")
           << outType.getDimSize(1) << ") and input (" << inType.getDimSize(1)
           << ")";
  }
  if (getLevel() <= 0) {
    return emitOpError(
               "mod_down_rescale is only valid for level > 0, got level ")
           << getLevel();
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError(
        "mod_down_rescale requires eval form for input and output");
  }
  return success();
}

// === ModDownEpilogueOp ===

LogicalResult ModDownEpilogueOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto nttBconvType = ::llvm::cast<MemRefType>(getNttBconv().getType());
  auto invPProdType = ::llvm::cast<MemRefType>(getInvPrimeProd().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, nttBconvType)) ||
      failed(verifyMemRefWordWidth(*this, invPProdType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outType, nttBconvType, "out", "ntt_bconv"))) {
    return failure();
  }
  if (invPProdType.getRank() != 1) {
    return emitOpError(
               "inv_prime_prod must be rank 1 (per-limb), but got rank ")
           << invPProdType.getRank();
  }
  if (invPProdType.getDimSize(0) != outType.getDimSize(0)) {
    return emitOpError("inv_prime_prod length (")
           << invPProdType.getDimSize(0) << ") must match output limb count ("
           << outType.getDimSize(0) << ")";
  }

  if (getPadStart() < 0 || getPadEnd() < getPadStart() ||
      getPadEnd() > outType.getDimSize(0)) {
    return emitOpError("pad range [") << getPadStart() << ", " << getPadEnd()
                                      << "] out of bounds for output with "
                                      << outType.getDimSize(0) << " limbs";
  }

  int64_t padLen = getPadEnd() - getPadStart();
  if (getSrc2()) {
    auto src2Type = ::llvm::cast<MemRefType>(getSrc2().getType());
    if (failed(verifyMemRefWordWidth(*this, src2Type))) return failure();
    if (src2Type.getRank() != 2) {
      return emitOpError("src2 must have rank 2, but got rank ")
             << src2Type.getRank();
    }
    if (src2Type.getDimSize(0) < padLen) {
      return emitOpError("src2 limb count (")
             << src2Type.getDimSize(0)
             << ") cannot be smaller than pad length (" << padLen << ")";
    }
    if (src2Type.getDimSize(1) != outType.getDimSize(1)) {
      return emitOpError("src2 degree (")
             << src2Type.getDimSize(1) << ") must match output degree ("
             << outType.getDimSize(1) << ")";
    }
  }

  if (getSrc2Padding()) {
    auto padType = ::llvm::cast<MemRefType>(getSrc2Padding().getType());
    if (failed(verifyMemRefWordWidth(*this, padType))) return failure();
    if (padType.getRank() != 1) {
      return emitOpError("src2_padding must be rank 1, but got rank ")
             << padType.getRank();
    }
  }

  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError(
        "mod_down_epilogue requires eval form for input and output");
  }

  return success();
}

// === NTTModDownOp ===

LogicalResult NTTModDownOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto nttInType = ::llvm::cast<MemRefType>(getNttIn().getType());
  auto invPProdType = ::llvm::cast<MemRefType>(getInvPrimeProd().getType());

  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, nttInType)) ||
      failed(verifyMemRefWordWidth(*this, invPProdType))) {
    return failure();
  }
  if (failed(verifySameShape(*this, outType, nttInType, "out", "ntt_in"))) {
    return failure();
  }
  if (invPProdType.getRank() != 1) {
    return emitOpError(
               "inv_prime_prod must be rank 1 (per-limb), but got rank ")
           << invPProdType.getRank();
  }
  if (invPProdType.getDimSize(0) != outType.getDimSize(0)) {
    return emitOpError("inv_prime_prod length (")
           << invPProdType.getDimSize(0) << ") must match output limb count ("
           << outType.getDimSize(0) << ")";
  }

  if (getPadStart() < 0 || getPadEnd() < getPadStart() ||
      getPadEnd() > outType.getDimSize(0)) {
    return emitOpError("pad range [") << getPadStart() << ", " << getPadEnd()
                                      << "] out of bounds for output with "
                                      << outType.getDimSize(0) << " limbs";
  }

  int64_t padLen = getPadEnd() - getPadStart();

  if (getSrc2Padding() && !getSrc2()) {
    return emitOpError("src2_padding requires src2");
  }

  if (getSrc2()) {
    auto src2Type = ::llvm::cast<MemRefType>(getSrc2().getType());
    if (failed(verifyMemRefWordWidth(*this, src2Type))) return failure();
    if (src2Type.getRank() != 2) {
      return emitOpError("src2 must have rank 2, but got rank ")
             << src2Type.getRank();
    }
    if (src2Type.getDimSize(0) < padLen + getSrc2Offset()) {
      return emitOpError("src2 limb count (")
             << src2Type.getDimSize(0)
             << ") cannot be smaller than pad length + src2_offset ("
             << padLen + getSrc2Offset() << ")";
    }
    if (src2Type.getDimSize(1) != outType.getDimSize(1)) {
      return emitOpError("src2 degree (")
             << src2Type.getDimSize(1) << ") must match output degree ("
             << outType.getDimSize(1) << ")";
    }
  }

  if (getSrc2Padding()) {
    auto padType = ::llvm::cast<MemRefType>(getSrc2Padding().getType());
    if (failed(verifyMemRefWordWidth(*this, padType))) return failure();
    if (padType.getRank() != 1) {
      return emitOpError("src2_padding must be rank 1, but got rank ")
             << padType.getRank();
    }
    if (padType.getDimSize(0) < padLen) {
      return emitOpError("src2_padding length (")
             << padType.getDimSize(0) << ") cannot be smaller than pad length ("
             << padLen << ")";
    }
  }

  if (getInForm() != Form::Coeff || getOutForm() != Form::Eval) {
    return emitOpError(
        "ntt_mod_down requires coeff form for input and eval form for output");
  }

  if (getPrimeIndices() && static_cast<int64_t>((*getPrimeIndices()).size()) !=
                               outType.getDimSize(0)) {
    return emitOpError("prime_indices size (")
           << (*getPrimeIndices()).size() << ") must match output limb count ("
           << outType.getDimSize(0) << ")";
  }

  return success();
}

// === KeySwitchNoModDownOp ===

LogicalResult KeySwitchNoModDownOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax"))) {
    return failure();
  }
  if (outBxType.getDimSize(1) != inType.getDimSize(1)) {
    return emitOpError("degree mismatch between output (")
           << outBxType.getDimSize(1) << ") and input (" << inType.getDimSize(1)
           << ")";
  }
  if (outBxType.getDimSize(0) <= inType.getDimSize(0)) {
    return emitOpError("output limb count (")
           << outBxType.getDimSize(0)
           << ") must be strictly greater than input limb count ("
           << inType.getDimSize(0) << ") to accommodate auxiliary primes";
  }
  if (getInBx()) {
    auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());
    if (failed(verifySameShape(*this, inType, inBxType, "in", "in_bx"))) {
      return failure();
    }
  }
  if (getInAx()) {
    auto inAxType = ::llvm::cast<MemRefType>(getInAx().getType());
    if (failed(verifySameShape(*this, inType, inAxType, "in", "in_ax"))) {
      return failure();
    }
  }
  if (failed(verifyEvkDigits(*this, getEvkBx(), getEvkAx(),
                             outBxType.getDimSize(1),
                             outBxType.getElementType()))) {
    return failure();
  }
  for (Value evk : getEvkBx()) {
    auto evkType = ::llvm::cast<MemRefType>(evk.getType());
    if (evkType.getDimSize(0) < outBxType.getDimSize(0)) {
      return emitOpError("evk digit limb count (")
             << evkType.getDimSize(0)
             << ") must be at least output limb count ("
             << outBxType.getDimSize(0) << ")";
    }
  }
  if (static_cast<int64_t>(getLevel()) < 0 &&
      (getLevel() != -1 || outBxType.getDimSize(0) != 2)) {
    return emitOpError("level must be non-negative");
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError(
        "key_switch_no_mod_down requires eval form for input and output");
  }
  return success();
}

// === KeySwitchOp ===

LogicalResult KeySwitchOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());
  auto inAxType = ::llvm::cast<MemRefType>(getInAx().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, inBxType)) ||
      failed(verifyMemRefWordWidth(*this, inAxType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(verifySameShape(*this, outBxType, inBxType, "out_bx", "in_bx")) ||
      failed(verifySameShape(*this, outBxType, inAxType, "out_bx", "in_ax"))) {
    return failure();
  }
  if (failed(verifyEvkDigits(*this, getEvkBx(), getEvkAx(),
                             outBxType.getDimSize(1),
                             outBxType.getElementType()))) {
    return failure();
  }
  if (static_cast<int64_t>(getLevel()) < 0 &&
      (getLevel() != -1 || outBxType.getDimSize(0) != 1)) {
    return emitOpError("level must be non-negative");
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("key_switch requires eval form for input and output");
  }
  return success();
}

// === RelinearizeOp ===

LogicalResult RelinearizeOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());
  auto inAxType = ::llvm::cast<MemRefType>(getInAx().getType());
  auto inRxType = ::llvm::cast<MemRefType>(getInRx().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, inBxType)) ||
      failed(verifyMemRefWordWidth(*this, inAxType)) ||
      failed(verifyMemRefWordWidth(*this, inRxType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(verifySameShape(*this, outBxType, inBxType, "out_bx", "in_bx")) ||
      failed(verifySameShape(*this, outBxType, inAxType, "out_bx", "in_ax")) ||
      failed(verifySameShape(*this, outBxType, inRxType, "out_bx", "in_rx"))) {
    return failure();
  }
  if (failed(verifyEvkDigits(*this, getEvkBx(), getEvkAx(),
                             outBxType.getDimSize(1),
                             outBxType.getElementType()))) {
    return failure();
  }
  if (static_cast<int64_t>(getLevel()) < 0) {
    return emitOpError("level must be non-negative, got ") << getLevel();
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("relinearize requires eval form for input and output");
  }
  return success();
}

// === RelinearizeRescaleOp ===

LogicalResult RelinearizeRescaleOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());
  auto inAxType = ::llvm::cast<MemRefType>(getInAx().getType());
  auto inRxType = ::llvm::cast<MemRefType>(getInRx().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, inBxType)) ||
      failed(verifyMemRefWordWidth(*this, inAxType)) ||
      failed(verifyMemRefWordWidth(*this, inRxType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(verifySameShape(*this, inBxType, inAxType, "in_bx", "in_ax")) ||
      failed(verifySameShape(*this, inBxType, inRxType, "in_bx", "in_rx"))) {
    return failure();
  }
  if (outBxType.getDimSize(1) != inBxType.getDimSize(1)) {
    return emitOpError("degree mismatch between output (")
           << outBxType.getDimSize(1) << ") and input ("
           << inBxType.getDimSize(1) << ")";
  }
  if (outBxType.getDimSize(0) >= inBxType.getDimSize(0)) {
    return emitOpError("output limb count (")
           << outBxType.getDimSize(0)
           << ") must be strictly less than input limb count ("
           << inBxType.getDimSize(0) << ") for relinearize_rescale";
  }
  if (failed(verifyEvkDigits(*this, getEvkBx(), getEvkAx(),
                             outBxType.getDimSize(1),
                             outBxType.getElementType()))) {
    return failure();
  }
  if (static_cast<int64_t>(getLevel()) <= 0) {
    return emitOpError(
               "relinearize_rescale is only valid for level > 0, got level ")
           << getLevel();
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError(
        "relinearize_rescale requires eval form for input and output");
  }
  return success();
}

// === HRotOp ===

LogicalResult HRotOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());
  auto inAxType = ::llvm::cast<MemRefType>(getInAx().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, inBxType)) ||
      failed(verifyMemRefWordWidth(*this, inAxType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(verifySameShape(*this, outBxType, inBxType, "out_bx", "in_bx")) ||
      failed(verifySameShape(*this, outBxType, inAxType, "out_bx", "in_ax"))) {
    return failure();
  }
  if (failed(verifyEvkDigits(*this, getEvkBx(), getEvkAx(),
                             outBxType.getDimSize(1),
                             outBxType.getElementType()))) {
    return failure();
  }
  if (static_cast<int64_t>(getLevel()) < 0) {
    return emitOpError("level must be non-negative, got ") << getLevel();
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("hrot requires eval form for input and output");
  }
  return success();
}

// === HConjOp ===

LogicalResult HConjOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());
  auto inAxType = ::llvm::cast<MemRefType>(getInAx().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, inBxType)) ||
      failed(verifyMemRefWordWidth(*this, inAxType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(verifySameShape(*this, outBxType, inBxType, "out_bx", "in_bx")) ||
      failed(verifySameShape(*this, outBxType, inAxType, "out_bx", "in_ax"))) {
    return failure();
  }
  if (failed(verifyEvkDigits(*this, getEvkBx(), getEvkAx(),
                             outBxType.getDimSize(1),
                             outBxType.getElementType()))) {
    return failure();
  }
  if (static_cast<int64_t>(getLevel()) < 0) {
    return emitOpError("level must be non-negative, got ") << getLevel();
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("hconj requires eval form for input and output");
  }
  return success();
}

// === HRotAddOp ===

LogicalResult HRotAddOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto in1BxType = ::llvm::cast<MemRefType>(getIn1Bx().getType());
  auto in1AxType = ::llvm::cast<MemRefType>(getIn1Ax().getType());
  auto in2BxType = ::llvm::cast<MemRefType>(getIn2Bx().getType());
  auto in2AxType = ::llvm::cast<MemRefType>(getIn2Ax().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, in1BxType)) ||
      failed(verifyMemRefWordWidth(*this, in1AxType)) ||
      failed(verifyMemRefWordWidth(*this, in2BxType)) ||
      failed(verifyMemRefWordWidth(*this, in2AxType))) {
    return failure();
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(
          verifySameShape(*this, outBxType, in1BxType, "out_bx", "in1_bx")) ||
      failed(
          verifySameShape(*this, outBxType, in1AxType, "out_bx", "in1_ax")) ||
      failed(
          verifySameShape(*this, outBxType, in2BxType, "out_bx", "in2_bx")) ||
      failed(
          verifySameShape(*this, outBxType, in2AxType, "out_bx", "in2_ax"))) {
    return failure();
  }
  if (failed(verifyEvkDigits(*this, getEvkBx(), getEvkAx(),
                             outBxType.getDimSize(1),
                             outBxType.getElementType()))) {
    return failure();
  }
  if (static_cast<int64_t>(getLevel()) < 0) {
    return emitOpError("level must be non-negative, got ") << getLevel();
  }
  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("hrot_add requires eval form for input and output");
  }
  return success();
}

// === ModUpToMaxOp ===

LogicalResult ModUpToMaxOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  auto inType = ::llvm::cast<MemRefType>(getIn().getType());
  if (failed(verifyMemRefWordWidth(*this, outType)) ||
      failed(verifyMemRefWordWidth(*this, inType))) {
    return failure();
  }
  if (outType.getRank() != 2 || inType.getRank() != 2) {
    return emitOpError("out and in must be rank-2 memrefs");
  }
  if (outType.getDimSize(1) != inType.getDimSize(1)) {
    return emitOpError("polynomial degree mismatch between output (")
           << outType.getDimSize(1) << ") and input (" << inType.getDimSize(1)
           << ")";
  }
  if (inType.getDimSize(0) < 1) {
    return emitOpError("in must have at least 1 limb");
  }
  if (getInForm() != Form::Coeff || getOutForm() != Form::Coeff) {
    return emitOpError(
        "mod_up_to_max requires coeff form for input and output");
  }
  int64_t numLimbs = outType.getDimSize(0);
  if (failed(verifyLimbRange(*this, getOutLimbRange(), numLimbs,
                             "out_limb_range"))) {
    return failure();
  }
  return verifyNumAux(*this, getNumAux(), numLimbs);
}

// === ZeroPadOp ===

LogicalResult ZeroPadOp::verify() {
  auto outType = ::llvm::cast<MemRefType>(getOut().getType());
  if (failed(verifyMemRefWordWidth(*this, outType))) return failure();
  if (outType.getRank() != 2) {
    return emitOpError("out must be a rank-2 memref");
  }
  ArrayRef<int64_t> range = getLimbRange();
  if (range.size() != 2) {
    return emitOpError("limb_range must have exactly 2 elements [start, end)");
  }
  int64_t start = range[0];
  int64_t end = range[1];
  int64_t numLimbs = outType.getDimSize(0);
  if (start < 0 || end < start || end > numLimbs) {
    return emitOpError("invalid limb_range [")
           << start << ", " << end << ") for buffer with " << numLimbs
           << " limbs";
  }
  return success();
}

// === HoistBSFusedOp ===

LogicalResult HoistBSFusedOp::verify() {
  size_t R = getOutBx().size();
  if (R < 1 || R > 8) {
    return emitOpError(
               "R (number of rotated outputs) must be in [1, 8], but got ")
           << R;
  }
  if (getOutAx().size() != R) {
    return emitOpError("out_ax count (")
           << getOutAx().size() << ") must match out_bx count (" << R << ")";
  }
  size_t beta = getModUp().size();
  if (beta < 1 || beta > 8) {
    return emitOpError(
               "beta (number of mod_up digits) must be in [1, 8], but got ")
           << beta;
  }

  auto firstOutBxType = ::llvm::cast<MemRefType>(getOutBx()[0].getType());
  if (failed(verifyMemRefWordWidth(*this, firstOutBxType))) return failure();
  if (!firstOutBxType.hasStaticShape()) {
    return emitOpError("out_bx buffers must have static shape");
  }
  if (firstOutBxType.getRank() != 2) {
    return emitOpError("out_bx must be rank-2 memrefs");
  }
  int64_t numLimbs = firstOutBxType.getDimSize(0);
  int64_t degree = firstOutBxType.getDimSize(1);
  Type elemType = firstOutBxType.getElementType();

  for (size_t i = 0; i < R; ++i) {
    auto bxType = ::llvm::cast<MemRefType>(getOutBx()[i].getType());
    auto axType = ::llvm::cast<MemRefType>(getOutAx()[i].getType());
    if (bxType.getRank() != 2 || bxType.getElementType() != elemType ||
        bxType.getDimSize(0) != numLimbs || bxType.getDimSize(1) != degree) {
      return emitOpError(
          "all out_bx buffers must have matching shape and element type");
    }
    if (axType.getRank() != 2 || axType.getElementType() != elemType ||
        axType.getDimSize(0) != numLimbs || axType.getDimSize(1) != degree) {
      return emitOpError(
          "all out_ax buffers must match out_bx shape and element type");
    }
  }

  for (size_t j = 0; j < beta; ++j) {
    auto modUpType = ::llvm::cast<MemRefType>(getModUp()[j].getType());
    if (modUpType.getRank() != 2 || modUpType.getElementType() != elemType ||
        modUpType.getDimSize(0) != numLimbs ||
        modUpType.getDimSize(1) != degree) {
      return emitOpError(
          "all mod_up buffers must match out_bx shape and element type");
    }
  }

  auto slabType = ::llvm::cast<MemRefType>(getEvkSlab().getType());
  if (slabType.getRank() != 1 || !slabType.getElementType().isInteger(8)) {
    return emitOpError("evk_slab must be a rank-1 memref of i8");
  }

  auto pseudoType = ::llvm::cast<MemRefType>(getInBxPseudoModup().getType());
  if (pseudoType.getRank() != 2 || pseudoType.getElementType() != elemType ||
      pseudoType.getDimSize(1) != degree) {
    return emitOpError(
        "in_bx_pseudo_modup must be rank-2 with matching degree and element "
        "type");
  }

  int64_t numQ = getNumQ();
  int64_t numAux = getNumAux();
  int64_t primeOffset = getPrimeOffset();
  if (numQ < 1) {
    return emitOpError("num_q must be >= 1, but got ") << numQ;
  }
  if (numAux < 0) {
    return emitOpError("num_aux must be >= 0, but got ") << numAux;
  }
  if (primeOffset < 0) {
    return emitOpError("prime_offset must be >= 0, but got ") << primeOffset;
  }
  if (numLimbs != numQ + numAux) {
    return emitOpError("output limb count (")
           << numLimbs << ") must equal num_q + num_aux (" << numQ + numAux
           << ")";
  }
  if (pseudoType.getDimSize(0) < numQ) {
    return emitOpError("in_bx_pseudo_modup limb count (")
           << pseudoType.getDimSize(0) << ") must be at least num_q (" << numQ
           << ")";
  }

  ArrayRef<int64_t> galois = getGaloisFactors();
  if (static_cast<size_t>(galois.size()) != R) {
    return emitOpError("galois_factors length (")
           << galois.size() << ") must equal R (" << R << ")";
  }

  ArrayRef<int64_t> evkNumQ = getEvkNumQ();
  if (static_cast<size_t>(evkNumQ.size()) != R) {
    return emitOpError("evk_num_q length (")
           << evkNumQ.size() << ") must equal R (" << R << ")";
  }
  for (size_t k = 0; k < R; ++k) {
    if (evkNumQ[k] < numQ) {
      return emitOpError("evk_num_q[") << k << "] (" << evkNumQ[k]
                                       << ") must be >= num_q (" << numQ << ")";
    }
  }

  ArrayRef<int64_t> evkOffsets = getEvkByteOffsets();
  if (static_cast<size_t>(evkOffsets.size()) != R &&
      static_cast<size_t>(evkOffsets.size()) != R * beta * 2) {
    return emitOpError("evk_byte_offsets length (")
           << evkOffsets.size() << ") must be either R (" << R
           << ") or R*beta*2 (" << R * beta * 2 << ")";
  }
  int64_t wordBytes = elemType.getIntOrFloatBitWidth() / 8;
  for (size_t i = 0; i < evkOffsets.size(); ++i) {
    int64_t off = evkOffsets[i];
    if (off < 0 || off % wordBytes != 0) {
      return emitOpError("evk_byte_offsets[")
             << i << "] (" << off << ") must be non-negative and a multiple of "
             << wordBytes;
    }
    if (slabType.hasStaticShape() && off >= slabType.getDimSize(0)) {
      return emitOpError("evk_byte_offsets[")
             << i << "] (" << off << ") exceeds evk_slab size ("
             << slabType.getDimSize(0) << ")";
    }
  }

  return success();
}

// === HoistGSFusedOp ===

LogicalResult HoistGSFusedOp::verify() {
  size_t G = getOutBx().size();
  if (G < 1 || G > 8) {
    return emitOpError(
               "G (number of giant-step outputs) must be in [1, 8], but got ")
           << G;
  }
  if (getOutAx().size() != G) {
    return emitOpError("out_ax count (")
           << getOutAx().size() << ") must match out_bx count (" << G << ")";
  }
  size_t B = getInBx().size();
  if (B < 1 || B > 8) {
    return emitOpError(
               "B (number of baby-step inputs) must be in [1, 8], but got ")
           << B;
  }
  if (getInAx().size() != B) {
    return emitOpError("in_ax count (")
           << getInAx().size() << ") must match in_bx count (" << B << ")";
  }

  auto firstOutBxType = ::llvm::cast<MemRefType>(getOutBx()[0].getType());
  if (failed(verifyMemRefWordWidth(*this, firstOutBxType))) return failure();
  if (!firstOutBxType.hasStaticShape()) {
    return emitOpError("out_bx buffers must have static shape");
  }
  if (firstOutBxType.getRank() != 2) {
    return emitOpError("out_bx must be rank-2 memrefs");
  }
  int64_t numLimbs = firstOutBxType.getDimSize(0);
  int64_t degree = firstOutBxType.getDimSize(1);
  Type elemType = firstOutBxType.getElementType();

  for (size_t g = 0; g < G; ++g) {
    auto bxType = ::llvm::cast<MemRefType>(getOutBx()[g].getType());
    auto axType = ::llvm::cast<MemRefType>(getOutAx()[g].getType());
    if (bxType.getRank() != 2 || bxType.getElementType() != elemType ||
        bxType.getDimSize(0) != numLimbs || bxType.getDimSize(1) != degree) {
      return emitOpError(
          "all out_bx buffers must have matching shape and element type");
    }
    if (axType.getRank() != 2 || axType.getElementType() != elemType ||
        axType.getDimSize(0) != numLimbs || axType.getDimSize(1) != degree) {
      return emitOpError(
          "all out_ax buffers must match out_bx shape and element type");
    }
  }

  for (size_t b = 0; b < B; ++b) {
    auto bxType = ::llvm::cast<MemRefType>(getInBx()[b].getType());
    auto axType = ::llvm::cast<MemRefType>(getInAx()[b].getType());
    if (bxType.getRank() != 2 || bxType.getElementType() != elemType ||
        bxType.getDimSize(0) != numLimbs || bxType.getDimSize(1) != degree) {
      return emitOpError(
          "all in_bx buffers must match out_bx shape and element type");
    }
    if (axType.getRank() != 2 || axType.getElementType() != elemType ||
        axType.getDimSize(0) != numLimbs || axType.getDimSize(1) != degree) {
      return emitOpError(
          "all in_ax buffers must match out_bx shape and element type");
    }
  }

  auto slabType = ::llvm::cast<MemRefType>(getConstsSlab().getType());
  if (slabType.getRank() != 1 || !slabType.getElementType().isInteger(8)) {
    return emitOpError("consts_slab must be a rank-1 memref of i8");
  }

  int64_t numQ = getNumQ();
  int64_t numAux = getNumAux();
  int64_t primeOffset = getPrimeOffset();
  int64_t ptLimbs = getPtLimbs();
  if (numQ < 1) {
    return emitOpError("num_q must be >= 1, but got ") << numQ;
  }
  if (numAux < 0) {
    return emitOpError("num_aux must be >= 0, but got ") << numAux;
  }
  if (primeOffset < 0) {
    return emitOpError("prime_offset must be >= 0, but got ") << primeOffset;
  }
  if (numLimbs != numQ + numAux) {
    return emitOpError("ciphertext limb count (")
           << numLimbs << ") must equal num_q + num_aux (" << numQ + numAux
           << ")";
  }
  if (ptLimbs < numLimbs) {
    return emitOpError("pt_limbs (")
           << ptLimbs << ") must be >= ciphertext limbs (" << numLimbs << ")";
  }

  ArrayRef<int64_t> ptOffsets = getPtWordOffsets();
  if (static_cast<size_t>(ptOffsets.size()) != G * B) {
    return emitOpError("pt_word_offsets length (")
           << ptOffsets.size() << ") must equal G * B (" << G * B << ")";
  }
  for (size_t i = 0; i < ptOffsets.size(); ++i) {
    if (ptOffsets[i] < -1) {
      return emitOpError("pt_word_offsets[")
             << i << "] (" << ptOffsets[i] << ") must be >= -1";
    }
  }

  return success();
}

// === EvalModOp ===

LogicalResult EvalModOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());
  auto inAxType = ::llvm::cast<MemRefType>(getInAx().getType());
  auto slabType = ::llvm::cast<MemRefType>(getEvkSlab().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, inBxType)) ||
      failed(verifyMemRefWordWidth(*this, inAxType))) {
    return failure();
  }
  if (!outBxType.hasStaticShape() || !inBxType.hasStaticShape()) {
    return emitOpError("operands must have static shape");
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(verifySameShape(*this, inBxType, inAxType, "in_bx", "in_ax"))) {
    return failure();
  }
  if (outBxType.getDimSize(1) != inBxType.getDimSize(1)) {
    return emitOpError("degree mismatch between output (")
           << outBxType.getDimSize(1) << ") and input ("
           << inBxType.getDimSize(1) << ")";
  }

  // eval_mod_start_level = 21 -> 22 limbs, eval_mod_end = 13 -> 14 limbs.
  int64_t inLimbs = inBxType.getDimSize(0);
  int64_t outLimbs = outBxType.getDimSize(0);
  if (inLimbs != 22) {
    return emitOpError("input limb count must be 22 (level 21), but got ")
           << inLimbs;
  }
  if (outLimbs != 14) {
    return emitOpError("output limb count must be 14 (level 13), but got ")
           << outLimbs;
  }

  if (slabType.getRank() != 1 || !slabType.getElementType().isInteger(8)) {
    return emitOpError("evk_slab must be a rank-1 memref of i8");
  }

  if (getRelinEvkByteOffset() < 0 || getRelinEvkByteOffset() % 8 != 0) {
    return emitOpError(
               "relin_evk_byte_offset must be >= 0 and a multiple of 8, but "
               "got ")
           << getRelinEvkByteOffset();
  }
  if (getRelinEvkNumQ() < 22) {
    return emitOpError("relin_evk_num_q must be >= 22, but got ")
           << getRelinEvkNumQ();
  }
  if (getRelinEvkBeta() <= 0) {
    return emitOpError("relin_evk_beta must be > 0, but got ")
           << getRelinEvkBeta();
  }

  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("eval_mod requires eval form for input and output");
  }

  return success();
}

// === HoistPhaseOp ===

LogicalResult HoistPhaseOp::verify() {
  auto outBxType = ::llvm::cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::cast<MemRefType>(getOutAx().getType());
  auto inBxType = ::llvm::cast<MemRefType>(getInBx().getType());
  auto inAxType = ::llvm::cast<MemRefType>(getInAx().getType());
  auto constsType = ::llvm::cast<MemRefType>(getConstsSlab().getType());
  auto evkType = ::llvm::cast<MemRefType>(getEvkSlab().getType());

  if (failed(verifyMemRefWordWidth(*this, outBxType)) ||
      failed(verifyMemRefWordWidth(*this, outAxType)) ||
      failed(verifyMemRefWordWidth(*this, inBxType)) ||
      failed(verifyMemRefWordWidth(*this, inAxType))) {
    return failure();
  }
  if (!outBxType.hasStaticShape() || !inBxType.hasStaticShape()) {
    return emitOpError("operands must have static shape");
  }
  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(verifySameShape(*this, inBxType, inAxType, "in_bx", "in_ax"))) {
    return failure();
  }
  if (outBxType.getDimSize(1) != inBxType.getDimSize(1)) {
    return emitOpError("degree mismatch between output (")
           << outBxType.getDimSize(1) << ") and input ("
           << inBxType.getDimSize(1) << ")";
  }

  if (constsType.getRank() != 1 || !constsType.getElementType().isInteger(8)) {
    return emitOpError("consts_slab must be a rank-1 memref of i8");
  }
  if (evkType.getRank() != 1 || !evkType.getElementType().isInteger(8)) {
    return emitOpError("evk_slab must be a rank-1 memref of i8");
  }

  StringRef kind = getKind();
  if (kind != "cts" && kind != "stc") {
    return emitOpError("kind must be 'cts' or 'stc', but got '") << kind << "'";
  }

  int64_t phase = getPhase();
  int64_t expectedInLimbs = 0;
  int64_t expectedOutLimbs = 0;
  if (kind == "cts") {
    if (phase < 0 || phase > 3) {
      return emitOpError("phase for 'cts' must be in [0, 3], but got ")
             << phase;
    }
    expectedInLimbs = 26 - phase;
    expectedOutLimbs = 25 - phase;
  } else {
    if (phase < 0 || phase > 2) {
      return emitOpError("phase for 'stc' must be in [0, 2], but got ")
             << phase;
    }
    expectedInLimbs = 14 - phase;
    expectedOutLimbs = 13 - phase;
  }

  int64_t inLimbs = inBxType.getDimSize(0);
  int64_t outLimbs = outBxType.getDimSize(0);
  if (inLimbs != expectedInLimbs) {
    return emitOpError("input limb count for ")
           << kind << " phase " << phase << " must be " << expectedInLimbs
           << ", but got " << inLimbs;
  }
  if (outLimbs != expectedOutLimbs) {
    return emitOpError("output limb count for ")
           << kind << " phase " << phase << " must be " << expectedOutLimbs
           << ", but got " << outLimbs;
  }

  int64_t numSlots = getNumSlots();
  int64_t maxSlots = std::min<int64_t>(32768, outBxType.getDimSize(1) / 2);
  if (numSlots < 256 || numSlots > maxSlots ||
      (numSlots & (numSlots - 1)) != 0) {
    return emitOpError("num_slots must be a power of 2 in [256, ")
           << maxSlots << "], but got " << numSlots;
  }

  return success();
}

// === BootOp ===

LogicalResult BootOp::verify() {
  auto outBxType = ::llvm::dyn_cast<MemRefType>(getOutBx().getType());
  auto outAxType = ::llvm::dyn_cast<MemRefType>(getOutAx().getType());
  auto inBxType = ::llvm::dyn_cast<MemRefType>(getInBx().getType());
  auto inAxType = ::llvm::dyn_cast<MemRefType>(getInAx().getType());
  auto constsType = ::llvm::dyn_cast<MemRefType>(getConstsSlab().getType());
  auto evkType = ::llvm::dyn_cast<MemRefType>(getEvkSlab().getType());

  if (!outBxType || !outAxType || !inBxType || !inAxType || !constsType ||
      !evkType) {
    return emitOpError("operands must be memrefs");
  }

  if (outBxType.getRank() != 2 || outAxType.getRank() != 2 ||
      inBxType.getRank() != 2 || inAxType.getRank() != 2) {
    return emitOpError("ciphertext buffers must be rank-2 memrefs");
  }

  if (!outBxType.hasStaticShape() || !inBxType.hasStaticShape()) {
    return emitOpError("ciphertext buffers must have static shape");
  }

  Type i64Type = IntegerType::get(getContext(), 64);
  if (outBxType.getElementType() != i64Type ||
      outAxType.getElementType() != i64Type ||
      inBxType.getElementType() != i64Type ||
      inAxType.getElementType() != i64Type) {
    return emitOpError("ciphertext buffers must have 64-bit integer elements");
  }

  if (failed(
          verifySameShape(*this, outBxType, outAxType, "out_bx", "out_ax")) ||
      failed(verifySameShape(*this, inBxType, inAxType, "in_bx", "in_ax"))) {
    return failure();
  }

  int64_t n = outBxType.getDimSize(1);
  if (n != inBxType.getDimSize(1)) {
    return emitOpError("degree mismatch between output (")
           << n << ") and input (" << inBxType.getDimSize(1) << ")";
  }

  int64_t inLimbs = inBxType.getDimSize(0);
  int64_t outLimbs = outBxType.getDimSize(0);
  if (inLimbs < 1 || inLimbs > 14) {
    return emitOpError("input limb count must be in [1, 14], but got ")
           << inLimbs;
  }
  if (outLimbs < 1 || outLimbs > 11) {
    return emitOpError("output limb count must be in [1, 11], but got ")
           << outLimbs;
  }

  int64_t numSlots = getNumSlots();
  int64_t maxSlots = std::min<int64_t>(32768, n / 2);
  if (numSlots < 256 || numSlots > maxSlots ||
      (numSlots & (numSlots - 1)) != 0) {
    return emitOpError("num_slots must be a power of 2 in [256, ")
           << maxSlots << "], but got " << numSlots;
  }

  if (constsType.getRank() != 1 || !constsType.getElementType().isInteger(8)) {
    return emitOpError("consts_slab must be a rank-1 memref of i8");
  }
  if (evkType.getRank() != 1 || !evkType.getElementType().isInteger(8)) {
    return emitOpError("evk_slab must be a rank-1 memref of i8");
  }

  if (getInForm() != Form::Eval || getOutForm() != Form::Eval) {
    return emitOpError("boot requires eval form for input and output");
  }

  return success();
}

}  // namespace cheddar_gpu
}  // namespace heir
}  // namespace mlir
