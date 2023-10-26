#ifndef INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_
#define INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_

include "TfheRustDialect.td"
include "TfheRustTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class TfheRust_Op<string mnemonic, list<Trait> traits = []> :
        Op<TfheRust_Dialect, mnemonic, traits> {

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type(results)
  }];
  let cppNamespace = "::mlir::heir::thfe_rs";
}


#endif  // INCLUDE_DIALECT_TFHERUST_IR_TFHERUSTOPS_TD_
