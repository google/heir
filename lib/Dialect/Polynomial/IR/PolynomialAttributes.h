#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_

#include "lib/Dialect/Polynomial/IR/Polynomial.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h.inc"

namespace mlir {
namespace heir {
namespace polynomial {

LogicalResult parseRingIntPolyAttr(AsmParser &parser, IntPolynomialAttr &attr);
void printRingIntPolyAttr(AsmPrinter &p, IntPolynomialAttr attr);

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALATTRIBUTES_H_
