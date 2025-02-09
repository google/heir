#ifndef LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_TD_
#define LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_TD_

include "LattigoDialect.td"
include "LattigoTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"

class Lattigo_Op<string mnemonic, list<Trait> traits = []> :
        Op<Lattigo_Dialect, mnemonic, traits # [OpAsmOpInterface]> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    // OpAsmOpInterface Methods
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
      for (auto result : getOperation()->getResults()) {
        if (auto ty = dyn_cast<TypeAsmInterface>(result.getType()))
          setNameFn(result, ty.suggestedName());
      }
    }
  }];
}

include "LattigoBGVOps.td"
include "LattigoCKKSOps.td"
include "LattigoRLWEOps.td"

#endif  // LIB_DIALECT_LATTIGO_IR_LATTIGOOPS_TD_
