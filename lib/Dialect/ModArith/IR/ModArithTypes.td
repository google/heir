#ifndef LIB_TYPES_MODARITH_IR_MODARITHTYPES_TD_
#define LIB_TYPES_MODARITH_IR_MODARITHTYPES_TD_

include "lib/Dialect/ModArith/IR/ModArithDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/AttrTypeBase.td"

class ModArith_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ModArith_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ModArith_ModArithType : ModArith_Type<"ModArith", "int", [MemRefElementTypeInterface]> {
  let summary = "Integer type with modular arithmetic";
  let description = [{
    `mod_arith.int<p>` represents an element of the ring of integers modulo $p$.
    The `modulus` attribute is the ring modulus, and `mod_arith` operations lower to
    `arith` operations that produce results in the range `[0, modulus)`, often called
    the _canonical representative_.

    `modulus` is specified with an integer type suffix, for example,
    `mod_arith.int<65537 : i32>`. This corresponds to the storage type for the
    modulus, and is `i64` by default.

    It is required that the underlying integer type should be larger than
    twice the modulus (have one extra bit of storage space) to avoid signedness
    issues. For example, when `modulus == 2 ** 32 - 1`, the underlying type
    for the modulus should be at least `i33`, though `i64` is a natural choice.

    Passes may allow intermediate values that do not always produce a
    canonical representative in `[0, modulus)`. For example, if the machine storage
    type is `i64`, but the `modulus` fits within an `i32`, a lowering could
    allow intermediate arithmetic values to grow to as large as an `i64` before
    reducing them. However, all passes must ensure that values used outside
    the local scope (e.g., function return values or arguments to calls to linked
    functions) are appropriately reduced to the canonical representative.
    `modulus` is the modulus the arithmetic working with.

    Examples:
    ```
    !Zp1 = !mod_arith.int<7> // implicitly being i64
    !Zp2 = !mod_arith.int<65537 : i32>
    !Zp3 = !mod_arith.int<536903681 : i64>
    ```
  }];
  let parameters = (ins
    "::mlir::IntegerAttr":$modulus
  );
  let assemblyFormat = "`<` $modulus `>`";
}

def ModArithLike: TypeOrValueSemanticsContainer<ModArith_ModArithType, "mod_arith-like">;

#endif  // LIB_TYPES_MODARITH_IR_MODARITHTYPES_TD_
