#include "lib/Dialect/RNS/IR/RNSAttributes.h"

#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

LogicalResult RNSAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<Attribute> values, RNSType type) {
  auto basisSize = type.getBasisTypes().size();
  if (values.size() != basisSize) {
    return emitError() << "expected " << basisSize
                       << " values to match the RNS basis size, but found "
                       << values.size();
  }
  return success();
}

void RNSAttr::print(AsmPrinter& printer) const {
  printer << "<[";
  // Use llvm::interleaveComma to handle the commas between elements nicely
  llvm::interleaveComma(getValues(), printer,
                        [&](Attribute attr) { printer << attr; });
  printer << "]>";
}

Attribute RNSAttr::parse(AsmParser& parser, Type type) {
  SmallVector<TypedAttr> attrs;

  if (parser.parseLess() || parser.parseLSquare()) return {};
  auto elementParser = [&]() {
    Attribute val;
    if (parser.parseAttribute(val)) return failure();
    if (auto typedVal = dyn_cast<TypedAttr>(val)) {
      attrs.push_back(typedVal);
      return success();
    }
    return failure();
  };
  if (parser.parseCommaSeparatedList(elementParser)) return {};
  if (parser.parseRSquare() || parser.parseGreater()) return {};

  // The rns type can be inferred from the types of the attribute values
  SmallVector<Type> basisTypes =
      map_to_vector(attrs, [](TypedAttr attr) { return attr.getType(); });
  RNSType rnsType = RNSType::get(parser.getContext(), basisTypes);
  SmallVector<Attribute> attrValues = map_to_vector(
      attrs, [](TypedAttr attr) { return cast<Attribute>(attr); });

  return RNSAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), ArrayRef<Attribute>(attrValues), rnsType);
}

// LimbwiseAttrInterface
::mlir::Attribute RNSAttr::assembleFromLimbs(
    ::mlir::Type resultAttrType, ::llvm::ArrayRef< ::mlir::Attribute> limbs) {
  auto rnsType = dyn_cast_if_present<RNSType>(resultAttrType);
  if (!rnsType) return {};
  return get(rnsType, limbs);
}

::mlir::Attribute RNSAttr::extractLimb(unsigned index) const {
  return getValues()[index];
}

::mlir::Type RNSAttr::getLimbType(unsigned index) const {
  return cast<RNSType>(getType()).getBasisTypes()[index];
}

unsigned RNSAttr::getNumLimbs() const { return getValues().size(); }

::mlir::Operation* RNSAttr::materializeConstant(::mlir::OpBuilder& builder,
                                                ::mlir::Location loc,
                                                ::mlir::Type type) const {
  auto rnsType = dyn_cast_if_present<RNSType>(type);
  if (!rnsType) return nullptr;
  auto op = rns::ConstantOp::create(builder, loc, rnsType, *this);
  return op.getOperation();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
