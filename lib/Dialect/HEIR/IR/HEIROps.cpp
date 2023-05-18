#include "include/Dialect/HEIR/IR/HEIROps.h"

#include "include/Dialect/HEIR/IR/HEIRDialect.h"
#include "mlir/include/mlir/IR/Dialect.h" // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h" // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h" // from @llvm-project

// Generated definitions.
#include "include/Dialect/HEIR/IR/HEIRDialect.cpp.inc"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// HEIR dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void HEIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "include/Dialect/HEIR/IR/HEIROps.cpp.inc"
      >();
  addTypes<CiphertextType>();
}

// Type parser for the dialect.
Type HEIRDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();

  if (keyword == "ciphertext") return CiphertextType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown `heir` type: ") << keyword;
  return {};
}

// Print a type registered to this dialect.
void HEIRDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (type.isa<CiphertextType>()) {
    os << "ciphertext";
    return;
  }

  llvm_unreachable("unexpected 'heir' type kind");
}

}  // namespace heir
}  // namespace mlir

// Ops definition from ODS

#define GET_OP_CLASSES
#include "include/Dialect/HEIR/IR/HEIROps.cpp.inc"
