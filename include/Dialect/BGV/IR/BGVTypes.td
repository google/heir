#ifndef HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_TD_
#define HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_TD_

include "BGVDialect.td"
include "BGVAttributes.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

// TODO(https://github.com/google/heir/issues/100): Add a plaintext type.

// A base class for all types in this dialect
class BGV_Type<string name, string typeMnemonic>
    : TypeDef<BGV_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Ciphertext : BGV_Type<"Ciphertext", "ciphertext"> {
  let summary = "A type for BGV Ciphertext";

  let description = [{
    A type for BGV Ciphertexts.

    This type tracks the BGV ciphertext parameters, including the ciphertext
    dimension (number of polynomials) and the set of rings that were used for
    the particular BGV scheme instance. The default dimension is 2, representing
    a ciphertext that is canonically encrypted against the key basis `(1, s)`.

    The type also includes a ring parameter specification.

    For example, `bgv.ciphertext<rings=#rings, dim=3>` is a ciphertext with 3
    polynomials (c_0, c_1, c_2).
  }];

  // TODO(https://github.com/google/heir/issues/99): Add # of plaintext bits.
  let parameters = (ins
    BGVRingArrayAttr:$rings,
    DefaultValuedParameter<"unsigned", "2">:$dim
  );

  let assemblyFormat = "`<` `rings` `=` $rings (`,` `dim` `=` $dim^ )? `>`";
}

#endif  // HEIR_INCLUDE_DIALECT_BGV_IR_BGVTYPES_TD_
