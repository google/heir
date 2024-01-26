#ifndef HEIR_INCLUDE_DIALECT_Swanky_IR_SwankyOPS_TD_
#define HEIR_INCLUDE_DIALECT_Swanky_IR_SwankyOPS_TD_

include "include/Dialect/Swanky/IR/SwankyDialect.td"


class Swanky_Op<string mnemonic, list<Trait> traits = []> :
        Op<Swanky_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::swanky";
}

#endif  // HEIR_INCLUDE_DIALECT_Swanky_IR_SwankyOPS_TD_
