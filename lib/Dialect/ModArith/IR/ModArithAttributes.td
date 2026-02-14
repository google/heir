#ifndef LIB_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_TD_

include "lib/Dialect/ModArith/IR/ModArithDialect.td"
include "lib/Dialect/ModArith/IR/ModArithTypes.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

class ModArith_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ModArith_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def ModArith_ModArithAttr : ModArith_Attr<"ModArith", "value"> {
  let summary = "a typed modular arithmetic value";
  let description = [{
    A typed modular arithmetic value.

    The `type` parameter is expected to be a modular arithmetic coefficient
    type, and `value` is the corresponding integer representative.

    Example:

    ```mlir
    #v = #mod_arith.value<17 : !mod_arith.int<...>>
    ```
  }];
  let parameters = (ins
    "::mlir::heir::mod_arith::ModArithType":$type,
    "::mlir::IntegerAttr":$value
  );
  let hasCustomAssemblyFormat = 1;
}

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_TD_
