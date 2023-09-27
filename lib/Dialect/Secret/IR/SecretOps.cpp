#include "include/Dialect/Secret/IR/SecretOps.h"

#include <memory>
#include <utility>

#include "include/Dialect/Secret/IR/SecretTypes.h"
#include "llvm/include/llvm/Support/Casting.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"               // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"    // from @llvm-project
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
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen()) return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
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

}  // namespace secret
}  // namespace heir
}  // namespace mlir
