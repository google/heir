#ifndef LIB_DIALECT_HEIRINTERFACES_H_
#define LIB_DIALECT_HEIRINTERFACES_H_

#include "mlir/include/mlir/IR/Builders.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"            // from @llvm-project

// IWYU pragma: begin_keep
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// Don't mess up order
#include "lib/Dialect/HEIRInterfaces.h.inc"
#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
// IWYU pragma: end_keep

namespace mlir {
namespace heir {

/// This class defines a Dialect Interface for any dialect wants to support
/// polynomial evaluation. The polynomial.eval op support an arbitrary input
/// type, and so this interface provides the necessary hooks to construct
/// multiplication and addition ops that work with that type, as well as
/// how to convert constants into the appropriate type.
class DialectPolynomialEvalInterface
    : public DialectInterface::Base<DialectPolynomialEvalInterface> {
 public:
  DialectPolynomialEvalInterface(Dialect *dialect) : Base(dialect) {}

  // Returns true if the dialect supports evaluation of the given polynomial
  // attribute. This would be false, for example, in the mod_arith dialect when
  // given a floating-point polynomial attribute as input.
  virtual bool supportsPolynomial(Attribute polynomialAttr) const = 0;

  // Construct one or more operations that convert the input integer or
  // floating point attribute to the appropriate type for this dialect. The
  // implementation should use the given OpBuilder to construct one or more
  // operations, and return a Value corresponding to the result of the
  // construction.
  //
  // Inputs:
  //
  //   - builder: the OpBuilder to use for construction
  //   - loc: the location to use for op construction
  //   - constantAttr: the constant attribute to convert
  //   - evaluatedType: the type of the input to the polynomial.eval op
  //
  //  Returns:
  //
  //    a Value corresponding to the result of the construction
  virtual Value constructConstant(OpBuilder &builder, Location loc,
                                  Attribute constantAttr,
                                  Type evaluatedType) const = 0;

  // FIXME: add add, mul, scalar_add, scalar_mul interface methods
};

// This is a convenience wrapper that collects all dialects implementing the
// DialectPolynomialEvalInterface, and provides a lookup from Operation* or
// Dialect* to the relevant interface.
class PolynomialEvalInterface
    : public DialectInterfaceCollection<DialectPolynomialEvalInterface> {
 public:
  using Base::Base;

  bool supportsPolynomial(Attribute polynomialAttr, Dialect *dialect);

  FailureOr<Value> constructConstant(OpBuilder &builder, Location loc,
                                     Attribute constantAttr, Type evaluatedType,
                                     Dialect *dialect);
};

void registerOperandAndResultAttrInterface(DialectRegistry &registry);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_HEIRINTERFACES_H_
