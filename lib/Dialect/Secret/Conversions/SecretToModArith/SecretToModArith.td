#ifndef LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_TD_
#define LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_TD_

include "mlir/Pass/PassBase.td"

def SecretToModArith : Pass<"secret-to-mod-arith"> {
  let summary = "Lower `secret` to `mod-arith` dialect.";

  let description = [{
    This pass lowers an IR with `secret.generic` blocks containing arithmetic
    operations to operations on plaintexts using the `mod_arith` dialect.
    This is primarily used in the plaintext lowering pipeline, where operations
    are performed directly against plaintexts.

    The pass assumes that the `secret.generic` regions have been distributed
    through arithmetic operations so that only one ciphertext operation appears
    per generic block. It also requires that `canonicalize` was run so that
    non-secret values used are removed from the `secret.generic`'s block
    arguments.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::mod_arith::ModArithDialect",
  ];

  let options = [
    Option<"plaintextModulus", "modulus", "int64_t",
           /*default=*/"0", "Modulus to use for the mod-arith dialect. "
           "If not specified, the pass will use the natural modulus for that integer type">,
    Option<"logScale", "log-scale", "int64_t",
           /*default=*/"0", "Log base 2 of the scale for encoding floating points as ints.">,
  ];
}

#endif  // LIB_DIALECT_SECRET_CONVERSIONS_SECRETTOMODARITH_SECRETTOMODARITH_TD_
