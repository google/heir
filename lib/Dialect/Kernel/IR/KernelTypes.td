#ifndef LIB_DIALECT_KERNEL_IR_KERNELTYPES_TD_
#define LIB_DIALECT_KERNEL_IR_KERNELTYPES_TD_

include "KernelDialect.td"

include "lib/Dialect/HEIRInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

class Kernel_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Kernel_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Kernel_PreparedLinearTransform : Kernel_Type<"PreparedLinearTransform",
    "prepared_linear_transform", [PlaintextTypeInterface]> {
  let summary = "A linear transform prepared for evaluation by a backend";
  let description = [{
    The result of `kernel.prepare_linear_transform`: the transform's diagonals
    encoded into plaintexts, ready to be applied to a ciphertext with
    `kernel.apply_linear_transform`.

    Note this op assumes the implementation kernel is semantically equivalent
    to Halevi-Shoup, aligning with most backend implementations.

    The type carries the static parameters:

    - `level`: the modulus level the diagonals are encoded at, which must equal
      the level of the ciphertext the transform is applied to. Backends are
      assumed to have a level-specific scaling factor.
    - `slots`: the ciphertext slot count the transform is encoded for.
    - `log_bsgs_ratio`: the log2 baby-step/giant-step ratio used to split the
      transform's rotations.

    The type implements PlaintextTypeInterface, so preprocessing
    dataflow analyses treat it like any other encoded plaintext.
  }];

  let parameters = (ins
    "int64_t":$level,
    "int64_t":$slots,
    "int64_t":$log_bsgs_ratio
  );
  let assemblyFormat = "`<` `level` `=` $level `,` `slots` `=` $slots `,` `log_bsgs_ratio` `=` $log_bsgs_ratio `>`";
}

#endif  // LIB_DIALECT_KERNEL_IR_KERNELTYPES_TD_
