#ifndef HEIR_INCLUDE_DIALECT_HEIR_IR_HEIRDIALECT_H_
#define HEIR_INCLUDE_DIALECT_HEIR_IR_HEIRDIALECT_H_

#include "mlir/include/mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h" // from @llvm-project

// Dialect main class is defined in ODS, we include it here. The
// constructor and the printing/parsing of dialect types are manually
// implemented (see ops.cpp).
#include "include/Dialect/HEIR/IR/HEIRDialect.h.inc"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// HEIR dialect types.
//===----------------------------------------------------------------------===//

// CiphertextType represents an encrypted value.
class CiphertextType
    : public Type::TypeBase<CiphertextType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace heir
}  // namespace mlir

#endif  // HEIR_INCLUDE_DIALECT_HEIR_IR_HEIRDIALECT_H_
