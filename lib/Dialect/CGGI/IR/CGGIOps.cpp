#include "lib/Dialect/CGGI/IR/CGGIOps.h"

#include <cstdint>
#include <optional>

#include "lib/Dialect/CGGI/IR/CGGIAttributes.h"
#include "lib/Dialect/CGGI/IR/CGGIEnums.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cggi {

std::optional<ValueRange> Lut2Op::getLookupTableInputs() {
  return ValueRange{getB(), getA()};
}

std::optional<ValueRange> Lut3Op::getLookupTableInputs() {
  return ValueRange{getC(), getB(), getA()};
}

std::optional<ValueRange> Lut4Op::getLookupTableInputs() {
  return ValueRange{getD(), getC(), getB(), getA()};
}

std::optional<ValueRange> LutLinCombOp::getLookupTableInputs() {
  return ValueRange{getInputs()};
}

std::optional<ValueRange> PackedLut3Op::getLookupTableInputs() {
  return std::nullopt;
}

LogicalResult LutLinCombOp::verify() {
  if (getInputs().size() != getCoefficients().size())
    return emitOpError("number of coefficients must match number of inputs");

  lwe::LWECiphertextType type = dyn_cast<lwe::LWECiphertextType>(
      getElementTypeOrSelf(getOutput().getType()));
  // Tablegen allows AnyType due to error using Variadic on TypeOrContainer
  // types.
  if (!type) return emitOpError("expected LWE ciphertext element type");
  auto plaintextBitwidth = type.getPlaintextSpace()
                               .getRing()
                               .getCoefficientType()
                               .getIntOrFloatBitWidth();

  int64_t maxCoeff = (1 << plaintextBitwidth) - 1;
  for (auto c : getCoefficients()) {
    if (c > maxCoeff) {
      InFlightDiagnostic diag =
          emitOpError("coefficient pushes error bits into message space");
      diag.attachNote() << "coefficient is " << c;
      diag.attachNote() << "largest allowable coefficient is " << maxCoeff;
      return diag;
    }
  }

  if (getLookupTable().getValue().getActiveBits() > maxCoeff + 1) {
    InFlightDiagnostic diag =
        emitOpError("LUT is larger than available cleartext bit width");
    diag.attachNote() << "LUT has "
                      << getLookupTable().getValue().getActiveBits()
                      << " active bits";
    diag.attachNote() << "max LUT size is " << maxCoeff + 1 << " bits";
    return diag;
  }

  return success();
}

LogicalResult ProgrammableBootstrapOp::verify() {
  lwe::LWECiphertextType type =
      cast<lwe::LWECiphertextType>(getElementTypeOrSelf(getOutput().getType()));
  auto plaintextBitwidth = type.getPlaintextSpace()
                               .getRing()
                               .getCoefficientType()
                               .getIntOrFloatBitWidth();

  int64_t maxCoeff = (1 << plaintextBitwidth) - 1;
  if (getLookupTable().getValue().getActiveBits() > maxCoeff + 1) {
    InFlightDiagnostic diag =
        emitOpError("LUT is larger than available cleartext bit width");
    diag.attachNote() << "LUT has "
                      << getLookupTable().getValue().getActiveBits()
                      << " active bits";
    diag.attachNote() << "max LUT size is " << maxCoeff + 1 << " bits";
    return diag;
  }

  return success();
}

LogicalResult MultiLutLinCombOp::verify() {
  if (getInputs().size() != getCoefficients().size())
    return emitOpError("number of coefficients must match number of inputs");
  if (getOutputs().size() != getLookupTables().size())
    return emitOpError("number of outputs must match number of LUTs");

  lwe::LWECiphertextType type =
      cast<lwe::LWECiphertextType>(getOutputs().front().getType());
  auto plaintextBitwidth = type.getPlaintextSpace()
                               .getRing()
                               .getCoefficientType()
                               .getIntOrFloatBitWidth();

  int64_t maxCoeff = (1 << plaintextBitwidth) - 1;
  for (auto c : getCoefficients()) {
    if (c > maxCoeff) {
      InFlightDiagnostic diag =
          emitOpError("coefficient pushes error bits into message space");
      diag.attachNote() << "coefficient is " << c;
      diag.attachNote() << "largest allowable coefficient is " << maxCoeff;
      return diag;
    }
  }

  for (int64_t lut : getLookupTables()) {
    APInt apintLut = APInt(64, lut);
    if (apintLut.getActiveBits() > maxCoeff + 1) {
      InFlightDiagnostic diag =
          emitOpError("LUT is larger than available cleartext bit width");
      diag.attachNote() << "LUT has " << apintLut.getActiveBits()
                        << " active bits";
      diag.attachNote() << "max LUT size is " << maxCoeff + 1 << " bits";
      return diag;
    }
  }

  return success();
}

FailureOr<PackedOp> buildBatchedBooleanGateOperation(
    MLIRContext* context, OpBuilder& builder, Operation* key,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  // Build gate list for the batched operation
  SmallVector<CGGIBoolGateEnumAttr> gateListAttrs;
  for (auto* op : batchedOperations) {
    FailureOr<CGGIBoolGateEnumAttr> attr =
        llvm::TypeSwitch<Operation&, FailureOr<CGGIBoolGateEnumAttr>>(*op)
            .Case<cggi::AndOp>([&context](AndOp op) {
              return CGGIBoolGateEnumAttr::get(context, CGGIBoolGateEnum::AND);
            })
            .Case<cggi::NandOp>([&context](NandOp op) {
              return CGGIBoolGateEnumAttr::get(context, CGGIBoolGateEnum::NAND);
            })
            .Case<cggi::XorOp>([&context](XorOp op) {
              return CGGIBoolGateEnumAttr::get(context, CGGIBoolGateEnum::XOR);
            })
            .Case<cggi::XNorOp>([&context](XNorOp op) {
              return CGGIBoolGateEnumAttr::get(context, CGGIBoolGateEnum::XNOR);
            })
            .Case<cggi::OrOp>([&context](OrOp op) {
              return CGGIBoolGateEnumAttr::get(context, CGGIBoolGateEnum::OR);
            })
            .Case<cggi::NorOp>([&context](NorOp op) {
              return CGGIBoolGateEnumAttr::get(context, CGGIBoolGateEnum::NOR);
            })
            .Default([&](Operation& op) -> FailureOr<CGGIBoolGateEnumAttr> {
              // Other operations are not supported for vectorization.
              return failure();
            });
    if (failed(attr)) {
      op->emitOpError("unsupported operation for vectorization");
      return failure();
    }
    gateListAttrs.push_back(attr.value());
  }
  auto boolGatesAttr = CGGIBoolGatesAttr::get(context, gateListAttrs);
  Type elementType = key->getResultTypes()[0];
  RankedTensorType resultTensorType = RankedTensorType::get(
      {static_cast<int64_t>(batchedOperations.size())}, elementType);
  return cggi::PackedOp::create(builder, key->getLoc(), resultTensorType,
                                boolGatesAttr, vectorizedOperands[0],
                                vectorizedOperands[1]);
}

bool isPackedGateOp(Operation* key, Operation* op) {
  return isa<AndOp, NandOp, XorOp, XNorOp, OrOp, NorOp>(op) &&
         key->getResultTypes() == op->getResultTypes() &&
         key->getAttrs() == op->getAttrs();
}

// BatchVectorizableOpInterface impl

bool AndOp::isBatchCompatible(Operation* rhs) {
  return isPackedGateOp(this->getOperation(), rhs);
}

FailureOr<Operation*> AndOp::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  return buildBatchedBooleanGateOperation(
      context, builder, this->getOperation(), vectorizedOperands,
      batchedOperations);
}

bool NandOp::isBatchCompatible(Operation* rhs) {
  return isPackedGateOp(this->getOperation(), rhs);
}

FailureOr<Operation*> NandOp::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  return buildBatchedBooleanGateOperation(
      context, builder, this->getOperation(), vectorizedOperands,
      batchedOperations);
}

bool NorOp::isBatchCompatible(Operation* rhs) {
  return isPackedGateOp(this->getOperation(), rhs);
}

FailureOr<Operation*> NorOp::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  return buildBatchedBooleanGateOperation(
      context, builder, this->getOperation(), vectorizedOperands,
      batchedOperations);
}

bool OrOp::isBatchCompatible(Operation* rhs) {
  return isPackedGateOp(this->getOperation(), rhs);
}

FailureOr<Operation*> OrOp::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  return buildBatchedBooleanGateOperation(
      context, builder, this->getOperation(), vectorizedOperands,
      batchedOperations);
}

bool XorOp::isBatchCompatible(Operation* rhs) {
  return isPackedGateOp(this->getOperation(), rhs);
}

FailureOr<Operation*> XorOp::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  return buildBatchedBooleanGateOperation(
      context, builder, this->getOperation(), vectorizedOperands,
      batchedOperations);
}

bool XNorOp::isBatchCompatible(Operation* rhs) {
  return isPackedGateOp(this->getOperation(), rhs);
}

FailureOr<Operation*> XNorOp::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  return buildBatchedBooleanGateOperation(
      context, builder, this->getOperation(), vectorizedOperands,
      batchedOperations);
}

bool NotOp::isBatchCompatible(Operation* rhs) {
  auto lhs = this->getOperation();
  return isa<NotOp>(rhs) && lhs->getAttrs() == rhs->getAttrs() &&
         lhs->getResultTypes() == rhs->getResultTypes();
}

FailureOr<Operation*> NotOp::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  Type elementType = this->getType();
  RankedTensorType resultTensorType = RankedTensorType::get(
      {static_cast<int64_t>(batchedOperations.size())}, elementType);
  Operation* op = cggi::NotOp::create(builder, this->getLoc(), resultTensorType,
                                      vectorizedOperands[0]);
  return op;
}

bool Lut3Op::isBatchCompatible(Operation* rhs) { return isa<Lut3Op>(rhs); }

FailureOr<Operation*> Lut3Op::buildBatchedOperation(
    MLIRContext* context, OpBuilder& builder,
    SmallVector<Value> vectorizedOperands,
    SmallVector<Operation*> batchedOperations) {
  SmallVector<Attribute> lutAttrs;
  for (auto* op : batchedOperations) {
    if (auto lut3Op = dyn_cast<Lut3Op>(op)) {
      lutAttrs.push_back(lut3Op.getLookupTable());
    } else {
      op->emitOpError("unsupported operation for vectorization");
      return failure();
    }
  }
  Type elementType = this->getType();
  RankedTensorType resultTensorType = RankedTensorType::get(
      {static_cast<int64_t>(batchedOperations.size())}, elementType);
  Operation* packedLut = cggi::PackedLut3Op::create(
      builder, this->getLoc(), resultTensorType, builder.getArrayAttr(lutAttrs),
      vectorizedOperands[0], vectorizedOperands[1], vectorizedOperands[2]);
  return packedLut;
}

}  // namespace cggi
}  // namespace heir
}  // namespace mlir
