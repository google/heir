#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/Polynomial.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "llvm/include/llvm/ADT/APInt.h"               // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"         // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"         // from @llvm-project
#include "mlir/include/mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::heir::polynomial;

#include "lib/Dialect/Polynomial/IR/PolynomialDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/Polynomial/IR/PolynomialOps.cpp.inc"

namespace mlir {
namespace heir {
namespace polynomial {

struct PolynomialOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Attribute, AliasResult>(attr)
                   .Case<RingAttr>([&](auto &ringAttr) {
                     os << "ring_";

                     auto coefficientType = ringAttr.getCoefficientType();
                     auto res =
                         llvm::TypeSwitch<Type, AliasResult>(coefficientType)
                             .Case<IntegerType>([&](auto &integerType) {
                               os << integerType;
                               return AliasResult::FinalAlias;
                             })
                             .template Case<mod_arith::ModArithType>(
                                 [&](auto &modArithType) {
                                   os << "Z";
                                   os << modArithType.getModulus().getValue();
                                   os << "_";
                                   os << modArithType.getModulus().getType();
                                   return AliasResult::FinalAlias;
                                 })
                             .Default([&](auto &type) {
                               return AliasResult::NoAlias;
                             });
                     if (res == AliasResult::NoAlias) {
                       return res;
                     }

                     auto polynomialModulus = ringAttr.getPolynomialModulus();
                     if (polynomialModulus) {
                       os << "_";
                       os << polynomialModulus.getPolynomial().toIdentifier();
                     }
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](auto &attr) { return AliasResult::NoAlias; });
    return res;
  }

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = TypeSwitch<Type, AliasResult>(type)
                   .Case<PolynomialType>([&](auto &polynomialType) {
                     os << "poly";
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](auto &type) { return AliasResult::NoAlias; });
    return res;
  }
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir

void PolynomialDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/Polynomial/IR/PolynomialOps.cpp.inc"
      >();

  addInterfaces<mlir::heir::polynomial::PolynomialOpAsmDialectInterface>();
}
