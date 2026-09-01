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
    assembly form the sentinels are written `R` (replication) and `G` (gap),
    e.g. `[R:4:1]`; the numeric ids are also accepted on input.
  }];

  let parameters = (ins
    "int64_t":$dim,
    "int64_t":$size,
    DefaultValuedParameter<"int64_t", "1">:$stride
  );

  let extraClassDeclaration = [{
    // Written `G`: slots this layout does not use.
    bool isGap() const { return getDim() == -2; }
    // Written `R`: the piece repeats the data instead of advancing through it.
    bool isReplicate() const { return getDim() == -1; }
    // Written `K`: the axis a matmul sums over. Only a `compute` placement
    // has one -- the result layout carries a plain gap in its place -- so
    // code walking a value's layout never meets it.
    bool isContraction() const { return getDim() == -3; }
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

    Optional **rolls** encode `roll(from, by)` metadata objects, applied left
    to right. Each argument is either a *piece* -- a position in the `dims`
    list, written as a bare integer -- or a whole tensor *axis*, written
    `axis N` (legal only when axis `N` is packed as more than one piece; the
    piece form is canonical for an unsplit axis, where the two coincide).

    ```mlir
    // Halevi-Shoup diagonal of a 4x4 matrix: slot j of ciphertext i holds
    // A[i, (j + i) mod 4]. Argument 1 (the axis-1 piece) is rolled by
    // argument 0 (the axis-0 piece).
    #diag = #rotom.layout<n = 4, rolls = [(1, 0)], dims = [[0:4:1] | [1:4:1]]>

    // Axis 1 is split across two pieces, so a whole-axis roll names its
    // argument `axis 1` rather than a piece position.
    #split = #rotom.layout<n = 16, rolls = [(axis 1, 2)],
                           dims = [[1:4:4], [1:4:1] | [0:16:1]]>
    ```

    When an axis is split, each piece reads one part of that axis's index:
    the piece of stride `s` and extent `e` reads `(i / s) mod e`, one place of
    that index in mixed radix. A piece FROM rewrites that
    part in place, `part(from) <- (part(from) - shift(by)) mod extent(from)`,
    leaving the axis's other pieces untouched. An `axis` FROM instead rewrites
    the whole axis index modulo its full extent and redistributes it over the
    pieces, so the subtraction can carry from one piece into the next.

    For example, axis 0 of extent 4 split across two pieces of extent 2 and
    replicated twice, so replica `d` supplies the shift `d`:

    ```mlir
    // `[0:2:2]` carries the high part of axis 0's index and `[0:2:1]` the low
    // part, so a replica's four slots hold elements 0, 1, 2, 3 in order. The
    // two layouts differ only in whether the roll names the axis or the piece
    // holding the low part.
    #by_axis = #rotom.layout<n = 8, rolls = [(axis 0, 0)],
                             dims = [[R:2:1], [0:2:2], [0:2:1]]>
    #by_piece = #rotom.layout<n = 8, rolls = [(2, 0)],
                              dims = [[R:2:1], [0:2:2], [0:2:1]]>
    ```

    Replica 0 shifts by 0, so both layouts hold `0, 1, 2, 3`. Replica 1 shifts
    by 1, and there they differ: `#by_axis` holds `1, 2, 3, 0`, the whole axis
    rotated by one, while `#by_piece` holds `1, 0, 3, 2`, each pair of slots
    rotated inside itself.

    The shift is the by argument's index: a
    traversal piece's part of its axis index (the whole index when the axis is
    unsplit), a whole axis's index via `axis`, a replication piece's replica
    index, or a gap piece's block index. The two extents need not match: the shift
    reduces modulo the rolled extent, so a smaller partner covers a prefix of
    the rotations and a larger one wraps.
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
