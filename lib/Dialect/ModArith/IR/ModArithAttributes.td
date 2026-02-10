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
    #v = #mod_arith.value<type=!mod_arith.int<17 : i32>, value=3 : i32>
    ```
  }];
  let parameters = (ins
    "::mlir::heir::mod_arith::ModArithType":$type,
    "::mlir::IntegerAttr":$value
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_MODARITH_IR_MODARITHATTRIBUTES_TD_
