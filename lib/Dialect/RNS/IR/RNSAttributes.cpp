#include "lib/Dialect/RNS/IR/RNSAttributes.h"

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "mlir/include/mlir/IR/Attributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"    // from @llvm-project

namespace mlir {
namespace heir {
namespace rns {

LogicalResult RNSAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<mlir::IntegerAttr> values,
    ::mlir::heir::rns::RNSType type) {
  auto basisSize = type.getBasisTypes().size();
  if (values.size() != basisSize) {
    return emitError() << "expected " << basisSize
                       << " values to match the RNS basis size, but found "
                       << values.size();
  }
  return success();
}

void RNSAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<[";
  // Use llvm::interleaveComma to handle the commas between elements nicely
  llvm::interleaveComma(getValues(), printer, [&](mlir::IntegerAttr attr) {
    printer << attr.getValue();
  });
  printer << "] : " << getType() << ">";
}

mlir::Attribute RNSAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  llvm::SmallVector<APInt> rawValues;
  RNSType rnsType;

  if (parser.parseLess() || parser.parseLSquare()) return {};
  // 1. Parse comma-separated integers: 3, 5, 7
  auto elementParser = [&]() {
    APInt val;
    if (parser.parseInteger(val)) return mlir::failure();
    rawValues.push_back(val);
    return mlir::success();
  };
  if (parser.parseCommaSeparatedList(elementParser)) return {};
  if (parser.parseRSquare() || parser.parseColon()) return {};
  if (parser.parseType(rnsType)) return {};
  if (parser.parseGreater()) return {};

  llvm::SmallVector<mlir::IntegerAttr> sizedValues;
  auto basisTypes = rnsType.getBasisTypes();
  for (auto [val, basisTy] : llvm::zip(rawValues, basisTypes)) {
    auto modArithTy = llvm::dyn_cast<mod_arith::ModArithType>(basisTy);
    if (!modArithTy) {
      parser.emitError(parser.getNameLoc())
          << "basis type is not a ModArithType";
      return {};
    }
    mlir::Type integerType = modArithTy.getModulus().getType();
    unsigned targetBitWidth = integerType.getIntOrFloatBitWidth();
    sizedValues.push_back(
        mlir::IntegerAttr::get(integerType, val.zextOrTrunc(targetBitWidth)));
  }

  return RNSAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), llvm::ArrayRef<mlir::IntegerAttr>(sizedValues),
      rnsType);
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
