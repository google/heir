#ifndef LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_TD_
#define LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_TD_

include "lib/Dialect/Rotom/IR/RotomDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"

class Rotom_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Rotom_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let genMnemonicAlias = 1;
}

def Rotom_DimAttr : Rotom_Attr<"Dim", "dim"> {
  let summary = "A single Rotom layout dimension.";
  let description = [{
    Encodes a traversal dimension: logical axis index, size, and stride.

    Sentinel `dim` values (no source tensor axis):

    * `-1`: replication (logical fill along this layout piece)
    * `-2`: gap (padding / unused slots; constrained to zero in materialization)

    Non-negative `dim` values index into the logical tensor shape. In the
    assembly form the sentinels are spelled `R` (replication) and `G` (gap),
    e.g. `[R:4:1]`; the numeric ids are also accepted on input.
  }];

  let parameters = (ins
    "int64_t":$dim,
    "int64_t":$size,
    DefaultValuedParameter<"int64_t", "1">:$stride
  );

  let extraClassDeclaration = [{
    bool isGap() const { return getDim() == -2; }
    bool isReplicate() const { return getDim() == -1; }
  }];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def Rotom_LayoutAttr : Rotom_Attr<"Layout", "layout"> {
  let summary = "A Rotom layout made from an ordered list of dims.";
  let description = [{
    A Rotom layout is an ordered list of `rotom.dim` dimensions plus the slot
    count `n` (ciphertext slot capacity).

    The verifier determines which dims map across ciphertexts vs within slots
    and checks the slot-side invariant that sizes and strides are powers of
    two (after splitting).

    For tensor_ext materialization, the **first** entry in `dims` is the
    ciphertext side of Rotom's `;` split (one piece); remaining entries are
    in-slot. See [Section 4.2 of the Rotom paper](https://eprint.iacr.org/2025/1319.pdf).

    Optional **rolls** encode a `roll(i,j)` metadata object: each pair `(i, j)`
    indexes into the `dims` array (the flattened `ct_dims + slot_dims` list) and
    rewrites `dims[i]`'s index to `(idx_i - s * idx_j) mod extent(dims[i])`,
    where the scale `s` defaults to 1 and may be spelled as a third tuple
    element `(i, j, s)`. The two extents need not match: the shift reduces
    modulo the rolled dim's extent, so a smaller partner covers a prefix of the
    rotations and a larger one wraps.

    When a traversal dim is packed as several mixed-radix pieces, a roll FROM
    any of its pieces rewrites the whole dim's index (each piece then takes its
    digit of the rolled index), and a roll BY one of its pieces shifts by that
    piece's digit -- together with a scale this expresses baby-step/giant-step
    diagonal packings.

    Optional **rollScales** stores one scale per roll pair; it must be omitted
    when every scale is 1 (the canonical form the parser and builders produce).
  }];

  let parameters = (ins
    "::mlir::ArrayAttr":$dims,
    "int64_t":$n,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$rolls,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$rollScales
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Layout with no `roll(i,j)` metadata (empty rolls storage).
    static ::mlir::heir::rotom::LayoutAttr get(::mlir::MLIRContext *context,
                                               ::mlir::ArrayAttr dims, int64_t n);
    /// Layout with unit-scale rolls (empty scale storage).
    static ::mlir::heir::rotom::LayoutAttr get(::mlir::MLIRContext *context,
                                               ::mlir::ArrayAttr dims, int64_t n,
                                               ::mlir::DenseI64ArrayAttr rolls);
  }];
}

def Rotom_SeedAttr : Rotom_Attr<"Seed", "seed"> {
  let summary = "A list of Rotom layouts for seeding.";
  let description = [{
    Contains a list of `rotom.layout` attributes representing the initial
    seeding layouts explored by Rotom.
  }];
  let parameters = (ins
    "::mlir::ArrayAttr":$layouts
  );

  let genVerifyDecl = 1;
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_TD_
