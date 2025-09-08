#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "lib/Utils/Layout/Codegen.h"
#include "llvm/include/llvm/ADT/DynamicAPInt.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/FlatLinearValueConstraints.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"         // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/ReshapeOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

using ::mlir::presburger::BoundType;
using ::mlir::presburger::IntegerRelation;
using ::mlir::presburger::VarKind;
using tensor_ext::NewLayoutAttr;

static std::string printRelation(const IntegerRelation& rel) {
  std::string str;
  llvm::raw_string_ostream os(str);
  rel.print(os);
  return str;
}

static FailureOr<Value> implementAssignLayoutNew(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  NewLayoutAttr layout = dyn_cast<NewLayoutAttr>(op.getLayout());
  if (!layout) {
    return op.emitError()
           << "Expected layout to be an IntegerRelation-style layout";
  }
  IntegerRelation rel = layout.getIntegerRelation();

  RankedTensorType dataSemanticType =
      cast<RankedTensorType>(op.getValue().getType());
  RankedTensorType ciphertextSemanticType = cast<RankedTensorType>(
      materializeNewLayout(dataSemanticType, layout, ciphertextSize));

  TypedValue<RankedTensorType> ciphertextTensor =
      cast<TypedValue<RankedTensorType>>(
          arith::ConstantOp::create(builder, ciphertextSemanticType,
                                    builder.getZeroAttr(ciphertextSemanticType))
              .getResult());

  MLIRLoopNestGenerator generator(builder);
  auto loopNestCstr = generateLoopNestAsCStr(rel);
  if (failed(loopNestCstr)) {
    return op.emitError() << "Failed to generate loop nest for relation "
                          << printRelation(rel);
  }
  LLVM_DEBUG(llvm::dbgs() << "Generating loop nest assignment for relation "
                          << loopNestCstr.value() << "\n");

  auto loop = generator.generateForLoop(
      rel, {ciphertextTensor},
      [&](mlir::OpBuilder& builder, Location loc, ValueRange exprs,
          ValueRange iterArgs) {
        // Extract from data and insert into ciphertextTensor
        auto extracted = builder.create<tensor::ExtractOp>(
            loc, op.getValue(),
            exprs.drop_back(ciphertextTensor.getType().getRank()));
        auto inserted = builder.create<tensor::InsertOp>(
            loc, extracted, iterArgs[0],
            exprs.drop_front(dataSemanticType.getRank()));
        return scf::ValueVector({inserted});
      });
  if (failed(loop)) {
    return op.emitError() << "Failed to generate loop nest for relation "
                          << printRelation(rel);
  }

  createdOpCallback(loop.value());
  return loop.value().getResults()[0];
}

static FailureOr<Value> implementUnpackOpNew(
    tensor_ext::UnpackOp op, ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  NewLayoutAttr layout = cast<NewLayoutAttr>(op.getLayout());
  IntegerRelation rel = layout.getIntegerRelation();

  MLIRLoopNestGenerator generator(builder);
  auto loopNestCstr = generateLoopNestAsCStr(rel);
  if (failed(loopNestCstr)) {
    return op.emitError() << "Failed to generate loop nest for relation "
                          << printRelation(rel);
  }
  LLVM_DEBUG(llvm::dbgs() << "Generating loop nest assignment for relation "
                          << loopNestCstr.value() << "\n");

  TypedValue<RankedTensorType> ciphertextTensor =
      cast<TypedValue<RankedTensorType>>(op.getValue());
  RankedTensorType dataSemanticType =
      cast<RankedTensorType>(op.getResult().getType());

  auto dataTensor = cast<TypedValue<RankedTensorType>>(
      mlir::arith::ConstantOp::create(builder,
                                      builder.getZeroAttr(dataSemanticType))
          .getResult());
  auto loop = generator.generateForLoop(
      rel, {dataTensor},
      [&](mlir::OpBuilder& builder, Location loc, ValueRange exprs,
          ValueRange iterArgs) {
        // Extract from ciphertext and insert into dataTensor
        auto extracted = builder.create<tensor::ExtractOp>(
            loc, ciphertextTensor,
            exprs.drop_front(dataSemanticType.getRank()));
        auto inserted = builder.create<tensor::InsertOp>(
            loc, extracted, iterArgs[0],
            exprs.drop_back(ciphertextTensor.getType().getRank()));
        return scf::ValueVector({inserted});
      });
  if (failed(loop)) {
    return op.emitError() << "Failed to generate loop nest for relation "
                          << printRelation(rel);
  }
  createdOpCallback(loop.value());
  return loop.value().getResults()[0];
}

FailureOr<Value> implementAssignLayout(
    tensor_ext::AssignLayoutOp op, int64_t ciphertextSize,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  OpBuilder::InsertionGuard guard(builder);
  // TODO(#2047): add a scalar version or augment scalar version below to
  // support new layout attr
  if (isa<NewLayoutAttr>(op.getLayout()) &&
      isa<RankedTensorType>(op.getResult().getType())) {
    return implementAssignLayoutNew(op, ciphertextSize, builder,
                                    createdOpCallback);
  }

  return failure();
};

FailureOr<Value> implementUnpackOp(
    tensor_ext::UnpackOp op, ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  OpBuilder::InsertionGuard guard(builder);
  // TODO(#2047): add a scalar version or augment scalar version below to
  // support new layout attr
  if (isa<NewLayoutAttr>(op.getLayout()) &&
      isa<RankedTensorType>(op.getResult().getType())) {
    return implementUnpackOpNew(op, builder, createdOpCallback);
  }

  return failure();
}

}  // namespace heir
}  // namespace mlir
