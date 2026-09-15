#ifndef LIB_DIALECT_ROTOM_IR_ROTOMOPS_TD_
#define LIB_DIALECT_ROTOM_IR_ROTOMOPS_TD_

include "lib/Dialect/Rotom/IR/RotomAttributes.td"
include "lib/Dialect/Rotom/IR/RotomDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Rotom_Op<string mnemonic, list<Trait> traits = []>
    : Op<Rotom_Dialect, mnemonic, traits>;

def Rotom_ConvertLayoutOp : Rotom_Op<"convert_layout",
    [Pure, AllTypesMatch<["input", "result"]>]> {
  let summary = "Convert from one Rotom layout to another.";
  let description = [{
    The op records one layout conversion decision. The `from` and `to`
    layouts pack the same tensor at the same ciphertext size `n`. The
    ciphertext counts of the two layouts can differ: an expansion and a
    compaction are both this op.

    The op stores the decision, not the steps. The lowering derives the
    rotate/mask/accumulate steps from the two layouts with
    `planLayoutConversion`, so the priced route is the emitted route.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       Rotom_LayoutAttr:$from,
                       Rotom_LayoutAttr:$to);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `:` type($input) `from` qualified($from) `to` qualified($to) attr-dict";
  let hasVerifier = 1;
}


def Rotom_ApplyRollOp : Rotom_Op<"apply_roll",
    [Pure, AllTypesMatch<["input", "result"]>]> {
  let summary = "Align a value by one roll of its layout.";
  let description = [{
    The op applies a roll transformation. The `to` layout is the `from`
    layout with exactly one more roll: `n` and the piece multiset are equal.
    The applied roll is the last roll of `to`. The rolled piece and its partner
    may trade places.

    A roll alignment is the cheap conversion of the Rotom paper (Section
    6.1). The lowering derives the rotation steps from the two layouts, the
    same way rotom.convert_layout does; the distinct op makes the alignment
    mechanism visible and separately priceable.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       Rotom_LayoutAttr:$from,
                       Rotom_LayoutAttr:$to);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `:` type($input) `from` qualified($from) `to` qualified($to) attr-dict";
  let hasVerifier = 1;
}

def Rotom_MatmulOp : Rotom_Op<"matmul", [Pure]> {
  let summary = "Matrix multiply of two layout-aligned operands.";
  let description = [{
    The compute kernel of a Rotom matmul plan. The operands arrive already
    aligned: `lhs_layout` and `rhs_layout` name their placements in the
    shared iteration space, `compute` names the placement the elementwise
    product computes at, and `to` names the result layout after the
    contraction axis is summed out. The lowering multiplies once at the
    compute placement and reduces the contraction axis from its position in
    `compute`: ciphertext-region pieces sum by block adds, slot-region
    pieces by a log-tree rotate-and-reduce.

    A matmul whose operand would arrive rolled by its ciphertext piece is a
    `rotom.bsgs_matmul` instead.
  }];
  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$rhs,
                       Rotom_LayoutAttr:$lhs_layout,
                       Rotom_LayoutAttr:$rhs_layout,
                       Rotom_LayoutAttr:$compute,
                       Rotom_LayoutAttr:$to);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs `:` functional-type(operands, results) attr-dict";
  let hasVerifier = 1;
}

def Rotom_BsgsMatmulOp : Rotom_Op<"bsgs_matmul", [Pure]> {
  let summary = "Matmul that folds an operand's roll into a BSGS schedule.";
  let description = [{
    A matmul whose contraction runs over a roll by the ciphertext piece, with
    the rolled operand facing a public one. Rolling first would cost one
    rotation per target ciphertext; folding the roll into the multiply costs
    B - 1 baby rotations, shared across the giant groups, plus G - 1 giant
    rotations (the reference's BSGS_ROT_ROLL + BSGS_MATMUL).

    The operand named by `roll_operand` (0 = lhs, 1 = rhs) arrives *unrolled*,
    at the layout its own `..._layout` attribute names; `rolled` is the layout
    the roll would have produced. Target ciphertext c reads that operand
    rotated by `roll_stride * c`, negated when `roll_negative`, for
    `roll_targets` targets. `baby` is B; G is roll_targets ceil-divided by it.

    ```mlir
    %0 = rotom.bsgs_matmul %w, %x : (tensor<10x512xf32>, tensor<512x1xf32>)
           -> tensor<10x1xf32>
      { roll_operand = 1 : i64, roll_stride = 64 : i64,
        roll_targets = 16 : i64, baby = 4 : i64, ... }
    ```
  }];
  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$rhs,
                       Rotom_LayoutAttr:$lhs_layout,
                       Rotom_LayoutAttr:$rhs_layout,
                       Rotom_LayoutAttr:$rolled,
                       Rotom_LayoutAttr:$compute,
                       Rotom_LayoutAttr:$to,
                       I64Attr:$roll_operand,
                       I64Attr:$roll_stride,
                       I64Attr:$roll_targets,
                       I64Attr:$baby,
                       UnitAttr:$roll_negative);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs `:` functional-type(operands, results) attr-dict";
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_ROTOM_IR_ROTOMOPS_TD_
