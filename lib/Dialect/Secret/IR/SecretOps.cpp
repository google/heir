#include "include/Dialect/Secret/IR/SecretOps.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>

#include "include/Dialect/Secret/IR/SecretPatterns.h"
#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Region.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace secret {

void YieldOp::print(OpAsmPrinter &p) {
  if (getNumOperands() > 0) p << ' ' << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs());
  if (getNumOperands() > 0) p << " : " << getOperandTypes();
}

ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> opInfo;
  SmallVector<Type, 2> types;
  SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

LogicalResult YieldOp::verify() {
  // Trait verifier ensures parent is a GenericOp
  auto parent = llvm::cast<GenericOp>(getParentOp());

  if (parent.getNumResults() != getNumOperands()) {
    return emitOpError()
           << "Expected yield op to have the same number of operands as its "
              "enclosing generic's result count. Yield had "
           << getNumOperands() << " operands, but enclosing generic op had "
           << parent.getNumResults() << " results.";
  }
  for (int i = 0; i < getValues().size(); ++i) {
    auto yieldSecretType = SecretType::get(getValues().getTypes()[i]);
    if (yieldSecretType != parent.getResultTypes()[i]) {
      return emitOpError()
             << "If a yield op returns types T, S, ..., then the enclosing "
                "generic op must have result types secret.secret<T>, "
                "secret.secret<S>, ... But this yield op has operand types: "
             << getValues().getTypes()
             << "; while the enclosing generic op has result types: "
             << parent.getResultTypes();
    }
  }
  return success();
}

void GenericOp::print(OpAsmPrinter &p) {
  ValueRange inputs = getInputs();
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";

  ArrayRef<NamedAttribute> attrs = (*this)->getAttrs();
  if (!attrs.empty()) {
    p << " attrs =";
    p.printOptionalAttrDict(attrs);
  }

  if (!getRegion().empty()) {
    p << ' ';
    p.printRegion(getRegion());
  }

  TypeRange resultTypes = getResults().getTypes();
  if (resultTypes.empty()) return;
  p.printOptionalArrowTypeList(resultTypes);
}

static ParseResult parseCommonStructuredOpParts(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<Type> &inputTypes) {
  SMLoc attrsLoc, inputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;

  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
      return failure();
  }
  attrsLoc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen()) return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  } else {
    inputsOperandsLoc = parser.getCurrentLocation();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

ParseResult GenericOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<Attribute> iteratorTypeAttrs;

  // Parsing is shared with named ops, except for the region.
  SmallVector<Type, 1> inputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes))
    return failure();

  // Optional attributes may be added.
  if (succeeded(parser.parseOptionalKeyword("attrs")))
    if (failed(parser.parseEqual()) ||
        failed(parser.parseOptionalAttrDict(result.attributes)))
      return failure();

  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parser.parseRegion(*region, {})) return failure();
  result.addRegion(std::move(region));

  SmallVector<Type, 1> outputTypes;
  if (parser.parseOptionalArrowTypeList(outputTypes)) return failure();
  result.addTypes(outputTypes);

  return success();
}

LogicalResult GenericOp::verify() {
  Block *body = getBody();

  // Verify that the operands of the body's basic block are the non-secret
  // analogues of the generic's operands.
  for (BlockArgument arg : body->getArguments()) {
    auto operand = getOperands()[arg.getArgNumber()];
    auto operandType = dyn_cast<SecretType>(operand.getType());

    // An error for the case where the generic operand and block arguments
    // are both non-secrets, but they are not the same type
    //
    // secret.generic (ins %x: i32) {
    //  ^bb0(%x_clear: i64):
    //   ...
    // }
    //
    if (!operandType && arg.getType() != operand.getType()) {
      return emitOpError()
             << "Type mismatch between block argument " << arg.getArgNumber()
             << " of type " << arg.getType() << " and generic operand of type "
             << operand.getType()
             << ". If the operand is not secret, it must be the same type as "
                "the block argument";
    }

    // An error for the case where the generic operand is secret,
    // but the corresponding block argument doesn't unwrap the secret.
    //
    // secret.generic (ins %x: !secret.secret<i32>) {
    //  ^bb0(%x_clear: i64):
    //   ...
    // }
    //
    if (operandType && arg.getType() != operandType.getValueType()) {
      return emitOpError()
             << "Type mismatch between block argument " << arg.getArgNumber()
             << " of type " << arg.getType() << " and generic operand of type "
             << operand.getType()
             << ". For a secret.secret<T> input to secret.generic, the "
                "corresponding block argument must have type T";
    }
  }

  return success();
}

void ConcealOp::build(OpBuilder &builder, OperationState &result,
                      Value cleartextValue) {
  Type resultType = SecretType::get(cleartextValue.getType());
  build(builder, result, resultType, cleartextValue);
}

void RevealOp::build(OpBuilder &builder, OperationState &result,
                     Value secretValue) {
  Type resultType =
      llvm::dyn_cast<SecretType>(secretValue.getType()).getValueType();
  build(builder, result, resultType, secretValue);
}

/// 'bodyBuilder' is used to build the body of secret.generic.
void GenericOp::build(OpBuilder &builder, OperationState &result,
                      ValueRange inputs, TypeRange outputTypes,
                      GenericOp::BodyBuilderFn bodyBuilder) {
  for (Type ty : outputTypes) result.addTypes(ty);
  result.addOperands(inputs);

  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  for (Value val : inputs) {
    SecretType secretType = dyn_cast<SecretType>(val.getType());
    Type blockType = secretType ? secretType.getValueType() : val.getType();
    bodyBlock.addArgument(blockType, val.getLoc());
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  bodyBuilder(builder, result.location, bodyBlock.getArguments());
}

OpFoldResult CastOp::fold(CastOp::FoldAdaptor adaptor) {
  Value input = getInput();
  Value output = getOutput();

  // self cast is a no-op
  if (input.getType() == output.getType()) {
    return input;
  }

  // Fold a cast-and-cast-back to a no-op.
  //
  //  %1 = secret.cast %0 : !secret.secret<T> to !secret.secret<U>
  //  %2 = secret.cast %1 : !secret.secret<U> to !secret.secret<T>
  //
  // folds to use %0 directly in place of %2.
  auto inputOp = input.getDefiningOp<CastOp>();
  if (!inputOp || output.getType() != inputOp.getInput().getType())
    return OpFoldResult();

  return inputOp.getInput();
}

OpOperand *GenericOp::getOpOperandForBlockArgument(Value value) {
  auto *body = getBody();
  int index = std::find(body->getArguments().begin(),
                        body->getArguments().end(), value) -
              body->getArguments().begin();
  if (index == body->getArguments().size()) return nullptr;

  return &getOperation()->getOpOperand(index);
}

std::optional<int> GenericOp::findResultIndex(Value value) {
  int index = std::find(getResults().begin(), getResults().end(), value) -
              getResults().begin();
  if (index < getNumResults()) return index;
  return std::nullopt;
}

YieldOp GenericOp::getYieldOp() {
  return *getBody()->getOps<YieldOp>().begin();
}

GenericOp cloneWithNewTypes(GenericOp op, TypeRange newTypes,
                            PatternRewriter &rewriter) {
  return rewriter.create<GenericOp>(
      op.getLoc(), op.getOperands(), newTypes,
      [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
        IRMapping mp;
        for (BlockArgument blockArg : op.getBody()->getArguments()) {
          mp.map(blockArg, blockArguments[blockArg.getArgNumber()]);
        }
        for (auto &op : op.getBody()->getOperations()) {
          b.clone(op, mp);
        }
      });
}

std::pair<GenericOp, ValueRange> GenericOp::addNewYieldedValues(
    ValueRange newValuesToYield, PatternRewriter &rewriter) {
  YieldOp yieldOp = getYieldOp();
  yieldOp.getValuesMutable().append(newValuesToYield);
  auto newTypes = llvm::to_vector<4>(
      llvm::map_range(yieldOp.getValues().getTypes(), [](Type t) -> Type {
        SecretType newTy = secret::SecretType::get(t);
        return newTy;
      }));
  GenericOp newOp = cloneWithNewTypes(*this, newTypes, rewriter);

  auto newResultStartIter = newOp.getResults().drop_front(
      newOp.getNumResults() - newValuesToYield.size());

  return {newOp, ValueRange(newResultStartIter)};
}

GenericOp GenericOp::removeYieldedValues(ValueRange yieldedValuesToRemove,
                                         PatternRewriter &rewriter,
                                         SmallVector<Value> &remainingResults) {
  YieldOp yieldOp = getYieldOp();
  for ([[maybe_unused]] Value v : yieldedValuesToRemove) {
    assert(llvm::is_contained(yieldOp.getValues(), v) &&
           "Cannot remove a value that is not yielded");
  }

  SmallVector<int, 4> indicesToErase;
  for (int i = 0; i < getYieldOp()->getNumOperands(); ++i) {
    if (std::find(yieldedValuesToRemove.begin(), yieldedValuesToRemove.end(),
                  getYieldOp()->getOperand(i)) != yieldedValuesToRemove.end()) {
      indicesToErase.push_back(i);
    } else {
      remainingResults.push_back(getResult(i));
    }
  }

  // Erase unused values in reverse to ensure deletion doesn't affect the next
  // indices to delete.
  for (int i : llvm::reverse(indicesToErase)) {
    getYieldOp().getValuesMutable().erase(i);
  }

  auto newResultTypes = llvm::to_vector<4>(
      llvm::map_range(yieldOp.getValues().getTypes(), [](Type t) -> Type {
        SecretType newTy = secret::SecretType::get(t);
        return newTy;
      }));

  return cloneWithNewTypes(*this, newResultTypes, rewriter);
}

void GenericOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<CollapseSecretlessGeneric, RemoveUnusedYieldedValues,
              RemoveUnusedGenericArgs, RemoveNonSecretGenericArgs>(context);
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir
