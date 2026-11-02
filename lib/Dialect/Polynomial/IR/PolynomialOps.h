#ifndef LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_
#define LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_

#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/Polynomial/IR/PolynomialTraits.h"
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "mlir/include/mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
// IWYU pragma: end_keep

#define GET_OP_CLASSES
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h.inc"

namespace mlir {
namespace heir {
namespace polynomial {

FailureOr<Value> buildApplyCoefficientwise(
    ImplicitLocOpBuilder& b, Value input, PolynomialType outputPolyTy,
    llvm::function_ref<FailureOr<Value>(ImplicitLocOpBuilder&, Value, Value)>
        transformCoeff);

// Builds a polynomial.apply_coefficientwise that selects the RNS limbs
// [start, start + size) of the input polynomial, preserving its form. The input
// must have an RNS coefficient type and the slice must be in bounds.
Value buildLimbSlice(ImplicitLocOpBuilder& b, Value input, IntegerAttr start,
                     IntegerAttr size);

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_POLYNOMIAL_IR_POLYNOMIALOPS_H_
