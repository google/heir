#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"
#include "lib/Utils/Layout/Codegen.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/DynamicAPInt.h"     // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/FlatLinearValueConstraints.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"           // from @llvm-project
#include "mlir/include/mlir/IR/OpDefinition.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"          // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"      // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"              // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

using ::mlir::presburger::IntegerRelation;
using tensor_ext::LayoutAttr;

static std::string printRelation(const IntegerRelation& rel) {
  std::string str;
  llvm::raw_string_ostream os(str);
  rel.print(os);
  return str;
}

static FailureOr<Value> implementAssignLayoutNew(
    Value input, LayoutAttr layout, int64_t ciphertextSize,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  IntegerRelation rel = layout.getIntegerRelation();

  RankedTensorType dataSemanticType =
      dyn_cast<RankedTensorType>(input.getType());
  RankedTensorType ciphertextSemanticType =
      cast<RankedTensorType>(materializeLayout(
          getElementTypeOrSelf(input.getType()), layout, ciphertextSize));
  if (!dataSemanticType) {
    // The input is a scalar, so we can just splat into the ciphertext tensor.
    // TODO(#2571): Validate layout is actually a fully dense layout.
    auto splatOp =
        tensor::SplatOp::create(builder, ciphertextSemanticType, input);
    createdOpCallback(splatOp);
    return splatOp.getResult();
  }

  // The input came from an empty tensor, so we can just create an empty
  // ciphertext semantic tensor type.
  if (auto emptyOp = dyn_cast_or_null<tensor::EmptyOp>(input.getDefiningOp())) {
    auto emptyCiphertextOp = tensor::EmptyOp::create(
        builder, builder.getLoc(), ciphertextSemanticType.getShape(),
        ciphertextSemanticType.getElementType());
    createdOpCallback(emptyCiphertextOp);
    return emptyCiphertextOp.getResult();
  }

  TypedValue<RankedTensorType> ciphertextTensor =
      cast<TypedValue<RankedTensorType>>(
          arith::ConstantOp::create(builder, ciphertextSemanticType,
                                    builder.getZeroAttr(ciphertextSemanticType))
              .getResult());

  MLIRLoopNestGenerator generator(builder);
  auto loopNestCstr = generateLoopNestAsCStr(rel);
  if (failed(loopNestCstr)) {
    return builder.emitError() << "Failed to generate loop nest for relation "
                               << printRelation(rel);
  }
  LLVM_DEBUG(llvm::dbgs() << "Generating loop nest assignment for relation "
                          << loopNestCstr.value() << "\n");

  auto loop = generator.generateForLoop(
      rel, {ciphertextTensor},
      [&](mlir::OpBuilder& builder, Location loc, ValueRange exprs,
          ValueRange iterArgs) {
        SmallVector<Value> extractIndices;
        for (Value idx :
             exprs.drop_back(ciphertextTensor.getType().getRank())) {
          extractIndices.push_back(arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), idx));
        }
        // Extract from data and insert into ciphertextTensor
        auto extracted =
            tensor::ExtractOp::create(builder, loc, input, extractIndices);

        SmallVector<Value> insertIndices;
        for (Value idx : exprs.drop_front(dataSemanticType.getRank())) {
          insertIndices.push_back(arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), idx));
        }
        auto inserted = tensor::InsertOp::create(builder, loc, extracted,
                                                 iterArgs[0], insertIndices);
        return scf::ValueVector({inserted});
      });
  if (failed(loop)) {
    return builder.emitError() << "Failed to generate loop nest for relation "
                               << printRelation(rel);
  }

  createdOpCallback(loop.value());
  return loop.value().getResults()[0];
}

static FailureOr<Value> implementUnpackOpNew(
    tensor_ext::UnpackOp op, ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  LayoutAttr layout = cast<LayoutAttr>(op.getLayout());
  IntegerRelation rel = layout.getIntegerRelation();

  RankedTensorType unpackedTensorType =
      dyn_cast<RankedTensorType>(op.getResult().getType());

  if (!unpackedTensorType) {
    // it's a scalar, so we can extract from any slot in the mapping
    std::vector<int64_t> point = anyRangePoint(rel);
    if (point.empty()) {
      return op.emitError()
             << "Failed to find any point in the range of relation "
             << printRelation(rel);
    }

    SmallVector<Value> indices;
    for (int64_t idx : point) {
      indices.push_back(arith::ConstantIndexOp::create(builder, idx));
    }
    auto extractOp = tensor::ExtractOp::create(builder, op.getValue(), indices);
    createdOpCallback(extractOp);
    return extractOp.getResult();
  }

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
        SmallVector<Value> extractIndices;
        for (Value idx : exprs.drop_front(dataSemanticType.getRank())) {
          extractIndices.push_back(arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), idx));
        }
        // Extract from ciphertext and insert into dataTensor
        auto extracted = tensor::ExtractOp::create(
            builder, loc, ciphertextTensor, extractIndices);
        SmallVector<Value> insertIndices;
        for (Value idx :
             exprs.drop_back(ciphertextTensor.getType().getRank())) {
          insertIndices.push_back(arith::IndexCastOp::create(
              builder, loc, builder.getIndexType(), idx));
        }
        auto inserted = tensor::InsertOp::create(builder, loc, extracted,
                                                 iterArgs[0], insertIndices);
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
    Value input, Attribute layout, int64_t ciphertextSize,
    ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  OpBuilder::InsertionGuard guard(builder);
  if (LayoutAttr layoutAttr = dyn_cast<LayoutAttr>(layout)) {
    return implementAssignLayoutNew(input, layoutAttr, ciphertextSize, builder,
                                    createdOpCallback);
  }
  return builder.emitError() << "Unsupported layout attribute type: " << layout;
};

FailureOr<Value> implementUnpackOp(
    tensor_ext::UnpackOp op, ImplicitLocOpBuilder& builder,
    const std::function<void(Operation*)>& createdOpCallback) {
  OpBuilder::InsertionGuard guard(builder);
  if (isa<LayoutAttr>(op.getLayout())) {
    return implementUnpackOpNew(op, builder, createdOpCallback);
  }

  return failure();
}

}  // namespace heir
}  // namespace mlir
