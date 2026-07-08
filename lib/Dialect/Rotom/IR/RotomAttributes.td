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

    Rotom's split between ciphertext dims and slot dims is denoted by a
    `|` inside the dims list (e.g., `dims = [[0:2:4] | [0:4:1]]`).
    Slot dims must fill `n` exactly; unused capacity is denoted
    with an explicit gap piece (e.g. `[G:4:1]`).
    See [Section 4.2 of the Rotom paper](https://eprint.iacr.org/2025/1319.pdf).

    Optional **rolls** encode a `roll(i,j)` metadata object: each pair `(i, j)`
    indexes into the `dims` array (the flattened `ct_dims + slot_dims` list) and
    uses modular addition to modify the indices of  `dims[i]` by the indices of
    `dims[j]`.
  }];

  let parameters = (ins
    "::mlir::ArrayAttr":$dims,
    "int64_t":$n,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$rolls
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Layout built from raw traversal dimensions.
    static ::mlir::heir::rotom::LayoutAttr getCanonical(
        ::mlir::MLIRContext *context, ::llvm::ArrayRef<DimAttr> dims,
        int64_t n, ::llvm::ArrayRef<int64_t> rolls = {});
    /// Canonicalizing builder with no `roll(from, by)` metadata.
    static ::mlir::heir::rotom::LayoutAttr get(::mlir::MLIRContext *context,
                                               ::mlir::ArrayAttr dims, int64_t n);
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
