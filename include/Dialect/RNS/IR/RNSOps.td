#ifndef INCLUDE_DIALECT_RNS_IR_RNSOPS_TD_
#define INCLUDE_DIALECT_RNS_IR_RNSOPS_TD_

include "include/Dialect/RNS/IR/RNSDialect.td"
include "include/Dialect/RNS/IR/RNSTypes.td"
include "mlir/IR/OpBase.td"

class RNS_Op<string mnemonic, list<Trait> traits = []> :
        Op<RNS_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::rns";
}

#endif  // INCLUDE_DIALECT_RNS_IR_RNSOPS_TD_
