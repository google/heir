#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_TD_

include "LattigoDialect.td"
include "LattigoTypes.td"
include "mlir/IR/OpBase.td"
include "lib/Utils/Tablegen/InPlaceOpInterface.td"

class Lattigo_Op<string mnemonic, list<Trait> traits = []> :
        Op<Lattigo_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
}

include "LattigoBGVOps.td"
include "LattigoCKKSOps.td"
include "LattigoRLWEOps.td"

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_TD_
