#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"

#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"       // from @llvm-project

namespace mlir {
namespace heir {
namespace mod_arith {

void ModArithAttr::print(mlir::AsmPrinter &printer) const {
  // Output: <1925 : !mod_arith.int<7681 : i32>>
  printer << "<" << getValue().getValue() << " : " << getType() << ">";
}

// syntax is <3 : type>
mlir::Attribute ModArithAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  APInt value;
  ModArithType modType;

  if (parser.parseLess()) return {};
  if (parser.parseInteger(value)) return {};
  if (parser.parseColon()) return {};
  if (parser.parseType(modType)) return {};
  if (parser.parseGreater()) return {};

  mlir::Type integerType = modType.getModulus().getType();
  unsigned targetBitWidth = integerType.getIntOrFloatBitWidth();
  value = value.zextOrTrunc(targetBitWidth);

  // 4. Construct the IntegerAttr using the element type of the ModArithType
  // This ensures the bitwidth of the value matches the modulus storage
  auto integerAttr = mlir::IntegerAttr::get(integerType, value);

  return ModArithAttr::get(parser.getContext(), modType, integerAttr);
}

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
