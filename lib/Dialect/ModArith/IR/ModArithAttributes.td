#ifndef LIB_DIALECT_MODARITH_IR_MODARITHATTRS_TD_
#define LIB_DIALECT_MODARITH_IR_MODARITHATTRS_TD_

include "lib/Dialect/ModArith/IR/ModArithDialect.td"
include "lib/Dialect/ModArith/IR/ModArithTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"

class ModArith_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<ModArith_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}


def ModArith_ModArithAttr : ModArith_Attr<
    "ModArith", "int", [TypedAttrInterface]> {
  let summary = "a typed mod_arith attribute";
  let description = [{
    Example:

    ```mlir
    #attr = 123:i32
    #attr_verbose = #mod_arith.int<123:i32>
    ```
  }];
  let parameters = (ins "::mlir::heir::mod_arith::ModArithType":$type, "mlir::IntegerAttr":$value);
  let assemblyFormat = "$value `:` $type";
  let builders = [
    AttrBuilderWithInferredContext<(ins "ModArithType":$type,
                                        "const int64_t":$value), [{
      return $_get(
        type.getContext(),
        type,
        IntegerAttr::get(type.getModulus().getType(), value));
    }]>
  ];
  let extraClassDeclaration = [{
    using ValueType = ::mlir::Attribute;
  }];
}


#endif  // LIB_DIALECT_MODARITH_IR_MODARITHATTRS_TD_
