#include "lib/Dialect/Secret/IR/SecretOps.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <utility>

#include "lib/Dialect/Secret/IR/SecretPatterns.h"
#include "lib/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"          // from @llvm-project
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

#define DEBUG_TYPE "secret-ops"

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
  for (size_t i = 0; i < getValues().size(); ++i) {
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
  // FIXME: why can't I just dyn_cast the Value to a BlockArgument?
  auto *body = getBody();
  int index = std::find(body->getArguments().begin(),
                        body->getArguments().end(), value) -
              body->getArguments().begin();
  if (index == (int)(body->getArguments().size())) return nullptr;

  return &getOperation()->getOpOperand(index);
}

std::optional<int> GenericOp::findResultIndex(Value value) {
  int index = std::find(getResults().begin(), getResults().end(), value) -
              getResults().begin();
  if (index < (int)getNumResults()) return index;
  return std::nullopt;
}

YieldOp GenericOp::getYieldOp() {
  return *getBody()->getOps<YieldOp>().begin();
}

GenericOp cloneWithNewResultTypes(GenericOp op, TypeRange newTypes,
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
  GenericOp newOp = cloneWithNewResultTypes(*this, newTypes, rewriter);

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
  for (unsigned int i = 0; i < getYieldOp()->getNumOperands(); ++i) {
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

  return cloneWithNewResultTypes(*this, newResultTypes, rewriter);
}

GenericOp GenericOp::removeYieldedValues(ArrayRef<int> yieldedIndicesToRemove,
                                         PatternRewriter &rewriter,
                                         SmallVector<Value> &remainingResults) {
  YieldOp yieldOp = getYieldOp();
  for ([[maybe_unused]] int index : yieldedIndicesToRemove) {
    assert(0 <= index && index < (int)yieldOp.getNumOperands() &&
           "Cannot remove an index that is out of range");
  }

  for (size_t i = 0; i < getYieldOp()->getNumOperands(); ++i) {
    if (std::find(yieldedIndicesToRemove.begin(), yieldedIndicesToRemove.end(),
                  i) == yieldedIndicesToRemove.end()) {
      remainingResults.push_back(getResult(i));
    }
  }

  // Erase unused values in reverse to ensure deletion doesn't affect the next
  // indices to delete.
  for (int i : llvm::reverse(yieldedIndicesToRemove)) {
    getYieldOp().getValuesMutable().erase(i);
  }

  auto newResultTypes = llvm::to_vector<4>(
      llvm::map_range(yieldOp.getValues().getTypes(), [](Type t) -> Type {
        SecretType newTy = secret::SecretType::get(t);
        return newTy;
      }));

  return cloneWithNewResultTypes(*this, newResultTypes, rewriter);
}

GenericOp GenericOp::extractOpBeforeGeneric(Operation *opToExtract,
                                            PatternRewriter &rewriter) {
  assert(opToExtract->getParentOp() == *this);
  LLVM_DEBUG({
    llvm::dbgs() << "Extracting:\n";
    opToExtract->dump();
  });

  // Result types are secret versions of the results of the op, since the
  // secret will yield all of this op's results immediately.
  SmallVector<Type> newResultTypes;
  newResultTypes.reserve(opToExtract->getNumResults());
  for (Type ty : opToExtract->getResultTypes()) {
    newResultTypes.push_back(SecretType::get(ty));
  }

  // The inputs to the new single-op generic are the subset of the current
  // generic's inputs that correspond to the opToExtract's operands, and any
  // operands among ops in opToExtract's nested regions.
  SmallVector<Value> newGenericOperands;
  SmallVector<Value> oldBlockArgs;
  DenseSet<Value> processedValues;
  newGenericOperands.reserve(opToExtract->getNumOperands());
  oldBlockArgs.reserve(opToExtract->getNumOperands());
  processedValues.reserve(opToExtract->getNumOperands());
  for (auto operand : opToExtract->getOperands()) {
    if (processedValues.count(operand)) continue;
    // If the yielded value is ambient, skip it and it continues to be ambient.
    auto *correspondingOperand = getOpOperandForBlockArgument(operand);
    if (!correspondingOperand) {
      // The operand must be ambient
      continue;
    }
    newGenericOperands.push_back(correspondingOperand->get());
    oldBlockArgs.push_back(operand);
    processedValues.insert(operand);
  }
  opToExtract->walk([&](Operation *nestedOp) {
    for (Value operand : nestedOp->getOperands()) {
      if (processedValues.count(operand)) continue;
      auto *correspondingOperand = getOpOperandForBlockArgument(operand);
      if (!correspondingOperand) {
        // Assume the operand is ambient, or else a block argument of
        // opToExtract or an op within a nested region of opToExtract.
        continue;
      }
      newGenericOperands.push_back(correspondingOperand->get());
      oldBlockArgs.push_back(operand);
      processedValues.insert(operand);
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "New single-op generic will have "
                          << newGenericOperands.size() << " operands\n");

  auto newGeneric = rewriter.create<GenericOp>(
      getLoc(), newGenericOperands, newResultTypes,
      [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
        IRMapping mp;
        // the newly-created blockArguments have the same index order as
        // newGenericOperands, which in turn shares the index ordering of
        // oldBlockArgs (they were constructed this way specifically to enable
        // this IR Mapping).
        for (auto [oldArg, newArg] : llvm::zip(oldBlockArgs, blockArguments)) {
          mp.map(oldArg, newArg);
        }
        auto *newOp = b.clone(*opToExtract, mp);
        b.create<YieldOp>(loc, newOp->getResults());
      });
  LLVM_DEBUG({
    llvm::dbgs() << "After adding new single-op generic:\n";
    newGeneric->getParentOp()->dump();
  });

  // Once the op is split off into a new generic op, we need to add new
  // operands to the old generic op, add new corresponding block arguments, and
  // replace all uses of the opToExtract's results with the created block
  // arguments.
  SmallVector<Value> oldGenericNewBlockArgs;
  rewriter.modifyOpInPlace(*this, [&]() {
    getInputsMutable().append(newGeneric.getResults());
    for (auto ty : opToExtract->getResultTypes()) {
      BlockArgument arg = getBody()->addArgument(ty, opToExtract->getLoc());
      oldGenericNewBlockArgs.push_back(arg);
    }
  });
  rewriter.replaceOp(opToExtract, oldGenericNewBlockArgs);

  return newGeneric;
}

void populateGenericCanonicalizers(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<CollapseSecretlessGeneric, RemoveUnusedYieldedValues,
               RemoveUnusedGenericArgs, RemoveNonSecretGenericArgs,
               HoistPlaintextOps>(ctx);
}

// When replacing a generic op with a new one, and given an op in the original
// generic op, find the corresponding op in the new generic op.
//
// Note, this is brittle and depends on the two generic ops having identical
// copies of the same ops in the same order.
Operation *findCorrespondingOp(GenericOp oldGenericOp, GenericOp newGenericOp,
                               Operation *op) {
  assert(oldGenericOp.getBody()->getOperations().size() ==
             newGenericOp.getBody()->getOperations().size() &&
         "findCorrespondingOp requires both oldGenericOp and newGenericOp have "
         "the same size");
  for (auto [oldOp, newOp] :
       llvm::zip(oldGenericOp.getBody()->getOperations(),
                 newGenericOp.getBody()->getOperations())) {
    if (&oldOp == op) {
      assert(oldOp.getName() == newOp.getName() &&
             "Expected corresponding op to be the same type in old and new "
             "generic");
      return &newOp;
    }
  }
  llvm_unreachable(
      "findCorrespondingOp used but no corresponding op was found");
  return nullptr;
}

std::pair<GenericOp, GenericOp> extractOpAfterGeneric(
    GenericOp genericOp, Operation *opToExtract, PatternRewriter &rewriter) {
  assert(opToExtract->getParentOp() == genericOp);
  [[maybe_unused]] auto *parent = genericOp->getParentOp();

  LLVM_DEBUG({
    llvm::dbgs() << "At start of extracting op after generic:\n";
    parent->dump();
  });
  // The new yields may not always be needed, and this can be cleaned up by
  // canonicalize, or a manual application of DedupeYieldedValues and
  // RemoveUnusedYieldedValues.
  auto result =
      genericOp.addNewYieldedValues(opToExtract->getOperands(), rewriter);
  // Can't do structured assignment of pair above, because clang fails to
  // compile the usage of these values in the closure below.
  // (https://stackoverflow.com/a/46115028/438830).
  GenericOp genericOpWithNewYields = result.first;
  ValueRange newResults = result.second;
  // Keep track of the opToExtract in the new generic.
  opToExtract =
      findCorrespondingOp(genericOp, genericOpWithNewYields, opToExtract);
  rewriter.replaceOp(genericOp,
                     ValueRange(genericOpWithNewYields.getResults().drop_back(
                         newResults.size())));
  LLVM_DEBUG({
    llvm::dbgs() << "After adding new yielded values:\n";
    parent->dump();
    llvm::dbgs() << "opToExtract is now in:\n";
    opToExtract->getParentOp()->dump();
  });

  SmallVector<Value> newGenericOperands;
  newGenericOperands.reserve(opToExtract->getNumOperands());
  for (auto operand : opToExtract->getOperands()) {
    // If the yielded value is a block argument or ambient, we can just use the
    // original SSA value.
    auto blockArg = mlir::dyn_cast<BlockArgument>(operand);
    bool isBlockArgOfGeneric =
        blockArg && blockArg.getOwner() == genericOpWithNewYields.getBody();
    bool isAmbient =
        (blockArg && blockArg.getOwner() != genericOpWithNewYields.getBody()) ||
        (!blockArg && operand.getDefiningOp()->getBlock() !=
                          genericOpWithNewYields.getBody());
    if (isBlockArgOfGeneric) {
      newGenericOperands.push_back(
          genericOpWithNewYields.getOperand(blockArg.getArgNumber()));
      continue;
    }
    if (isAmbient) {
      newGenericOperands.push_back(operand);
      continue;
    }

    // Otherwise, find the corresponding result of the generic op
    auto yieldOperands = genericOpWithNewYields.getYieldOp().getOperands();
    int resultIndex =
        std::find(yieldOperands.begin(), yieldOperands.end(), operand) -
        yieldOperands.begin();
    newGenericOperands.push_back(genericOpWithNewYields.getResult(resultIndex));
  }

  // Result types are secret versions of the results of the op, since the
  // secret will yield all of this op's results immediately.
  SmallVector<Type> newResultTypes;
  newResultTypes.reserve(opToExtract->getNumResults());
  for (Type ty : opToExtract->getResultTypes()) {
    newResultTypes.push_back(SecretType::get(ty));
  }

  rewriter.setInsertionPointAfter(genericOpWithNewYields);
  auto newGeneric = rewriter.create<GenericOp>(
      genericOpWithNewYields.getLoc(), newGenericOperands, newResultTypes,
      [&](OpBuilder &b, Location loc, ValueRange blockArguments) {
        IRMapping mp;
        int i = 0;
        for (Value operand : opToExtract->getOperands()) {
          mp.map(operand, blockArguments[i]);
          ++i;
        }
        auto *newOp = b.clone(*opToExtract, mp);
        b.create<YieldOp>(loc, newOp->getResults());
      });
  LLVM_DEBUG({
    llvm::dbgs() << "After adding new single-op generic:\n";
    parent->dump();
  });

  // Once the op is split off into a new generic op, we need to erase
  // the old op and remove its results from the yield op.
  rewriter.setInsertionPointAfter(genericOpWithNewYields);
  SmallVector<Value> remainingResults;
  auto replacedGeneric = genericOpWithNewYields.removeYieldedValues(
      opToExtract->getResults(), rewriter, remainingResults);
  // Keep track of the opToExtract in the new generic.
  opToExtract =
      findCorrespondingOp(genericOpWithNewYields, replacedGeneric, opToExtract);
  rewriter.replaceAllUsesWith(remainingResults, replacedGeneric.getResults());
  rewriter.eraseOp(genericOpWithNewYields);
  rewriter.eraseOp(opToExtract);
  LLVM_DEBUG({
    llvm::dbgs() << "After removing opToExtract from old generic:\n";
    parent->dump();
  });

  return std::pair{replacedGeneric, newGeneric};
}

void GenericOp::inlineInPlaceDroppingSecrets(PatternRewriter &rewriter,
                                             ValueRange operands) {
  GenericOp &op = *this;
  Block *originalBlock = op->getBlock();
  Block &opEntryBlock = op.getRegion().front();
  YieldOp yieldOp = dyn_cast<YieldOp>(op.getRegion().back().getTerminator());

  // Inline the op's (unique) block, including the yield op. This also
  // requires splitting the parent block of the generic op, so that we have a
  // clear insertion point for inlining.
  Block *newBlock = rewriter.splitBlock(originalBlock, Block::iterator(op));
  rewriter.inlineRegionBefore(op.getRegion(), newBlock);

  // Now that op's region is inlined, the operands of its YieldOp are mapped
  // to the materialized target values. Therefore, we can replace the op's
  // uses with those of its YieldOp's operands.
  rewriter.replaceOp(op, yieldOp->getOperands());

  // No need for these intermediate blocks, merge them into 1.
  rewriter.mergeBlocks(&opEntryBlock, originalBlock, operands);
  rewriter.mergeBlocks(newBlock, originalBlock, {});

  rewriter.eraseOp(yieldOp);
}

void GenericOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  populateGenericCanonicalizers(results, context);
}

}  // namespace secret
}  // namespace heir
}  // namespace mlir
