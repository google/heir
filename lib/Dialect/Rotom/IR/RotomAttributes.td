#ifndef LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_TD_
#define LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_TD_

include "lib/Dialect/Rotom/IR/RotomDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"

class Rotom_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Rotom_Dialect, name, traits # [OpAsmAttrInterface]> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # attrMnemonic # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def Rotom_DimAttr : Rotom_Attr<"Dim", "dim"> {
  let summary = "A single Rotom layout dimension.";
  let description = [{
    Encodes a traversal dimension: logical axis index, size, and stride.

    Sentinel `dim` values (no source tensor axis):

    * `-1`: replication (logical fill along this layout piece)
    * `-2`: gap (padding / unused slots; constrained to zero in materialization)

    Non-negative `dim` values index into the logical tensor shape.
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
    uses modular addition to modify the indices of  `dims[i]` by the indices of
    `dims[j]`.
  }];

  let parameters = (ins
    "::mlir::ArrayAttr":$dims,
    "int64_t":$n,
    OptionalParameter<"::mlir::DenseI64ArrayAttr">:$rolls
  );

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Layout with no `roll(i,j)` metadata (empty rolls storage).
    static ::mlir::heir::rotom::LayoutAttr get(::mlir::MLIRContext *context,
                                               ::mlir::ArrayAttr dims, int64_t n);
  }];
}

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMATTRIBUTES_TD_
