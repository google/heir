#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_TD_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_TD_

include "BGVDialect.td"
include "BGVAttributes.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

// TODO(#100): Add a plaintext type.

// A base class for all types in this dialect
class BGV_Type<string name, string typeMnemonic>
    : TypeDef<BGV_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Ciphertext : BGV_Type<"Ciphertext", "ciphertext"> {
  let summary = "a BGV ciphertext";

  let description = [{
    This type tracks the BGV ciphertext parameters, including the ciphertext
    dimension (number of polynomials) and the set of rings that were used for
    the particular BGV scheme instance. The default dimension is 2, representing
    a ciphertext that is canonically encrypted against the key basis $(1, s)$.

    The type also includes a ring parameter specification.

    For example, `bgv.ciphertext<rings=#rings, dim=3>` is a ciphertext with 3
    polynomials $(c_0, c_1, c_2)$.

    The optional attribute `level` specifies the "current ring".
  }];

  // TODO(#99): Add # of plaintext bits.
  let parameters = (ins
    BGVRingArrayAttr:$rings,
    DefaultValuedParameter<"unsigned", "2">:$dim,
    OptionalParameter<"std::optional<uint64_t>">:$level
  );

  let assemblyFormat = "`<` `rings` `=` $rings (`,` `dim` `=` $dim^ )? (`,` `level` `=` $level^ )? `>`";
}

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_TD_
