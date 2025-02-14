#include "lib/Dialect/RNS/IR/RNSDialect.h"

#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

// NOLINTNEXTLINE(misc-include-cleaner): Required to define RNSOps
#include "lib/Dialect/RNS/IR/RNSOps.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"

// Generated definitions
#include "lib/Dialect/RNS/IR/RNSDialect.cpp.inc"
#include "mlir/include/mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/RNS/IR/RNSTypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/RNS/IR/RNSOps.cpp.inc"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.cpp.inc"

namespace mlir {
namespace heir {
namespace rns {

struct RNSOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const override {
    auto res = llvm::TypeSwitch<Type, AliasResult>(type)
                   .Case<RNSType>([&](auto &rnsType) {
                     os << "rns";
                     auto size = rnsType.getBasisTypes().size();
                     os << "_L";
                     os << size - 1;  // start from 0
                     return AliasResult::FinalAlias;
                   })
                   .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }
};

void RNSDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/RNS/IR/RNSTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/RNS/IR/RNSOps.cpp.inc"
      >();

  addInterface<RNSOpAsmDialectInterface>();
}

}  // namespace rns
}  // namespace heir
}  // namespace mlir
