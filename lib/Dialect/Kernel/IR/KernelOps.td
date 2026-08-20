#ifndef LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_
#define LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_

include "KernelDialect.td"
include "KernelTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "lib/Dialect/HEIRInterfaces.td"

class Kernel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Kernel_Dialect, mnemonic, traits # [Pure]> {
  let cppNamespace = "::mlir::heir::kernel";
}

def Kernel_EvalChebyshevOp : Kernel_Op<"eval_chebyshev", [
  Pure,
  DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop", "getOperandsToReduce"]>,
  ElementwiseMappable,
  IncreasesMulDepthOpInterface
]> {
  let summary = "Evaluates a Chebyshev polynomial.";
  let description = [{
    The `kernel.eval_chebyshev` operation evaluates a Chebyshev polynomial
    on a given input. The domain is required to be `[-1, 1]`.
  }];

  let arguments = (ins
    AnyType:$input,
    ArrayAttr:$coefficients
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Kernel_LinearTransformOp : Kernel_Op<"linear_transform", [
  Pure,
  DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop", "getOperandsToReduce"]>,
  IncreasesMulDepthOpInterface
]> {
  let summary = "Performs a linear transform (matrix-vector multiplication) on a ciphertext.";
  let description = [{
    Performs a linear transform on a ciphertext using the diagonal method. Row
    k of the `diagonals` operand is the generalized diagonal named by
    `diagonal_indices[k]`. If `source_row_indices` is present, that diagonal is
    instead read from row `source_row_indices[k]`.
  }];

  let arguments = (ins
    AnyType:$input,
    // The generalized diagonals of the matrix, as a value rather than an
    // attribute: they are often produced by preprocessing (packing the weights
    // once, ahead of any query) rather than folded at compile time. A consumer
    // that needs the values at compile time can recover them with m_Constant.
    // This also matches the backend ops this lowers to, whose diagonals are
    // likewise operands.
    AnyType:$diagonals,
    DenseI64ArrayAttr:$diagonal_indices,
    OptionalAttr<DenseI64ArrayAttr>:$source_row_indices,
    OptionalAttr<F64Attr>:$bsgs_ratio
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = "$input `,` $diagonals attr-dict `:` type($input) `,` type($diagonals) `->` type($output)";
  let hasVerifier = 1;
}

def Kernel_PrepareLinearTransformOp : Kernel_Op<"prepare_linear_transform", [
  Pure,
  PlaintextEncodeOpInterface
]> {
  let summary = "Prepares (encodes) a linear transform for later application.";
  let description = [{
    Encodes the generalized diagonals of a matrix into a backend-specific
    prepared transform, at the level, slot count, and BSGS split recorded in
    the result type. The prepared transform is applied to a ciphertext with
    `kernel.apply_linear_transform`.

    Preparation is pure cleartext work (row selection, float conversion,
    plaintext encoding), so the op implements PlaintextEncodeOpInterface and
    split-preprocessing hoists it out of the hot path: an inference then only
    pays for the application.

    As with `kernel.linear_transform`, row k of the `diagonals` operand is the
    generalized diagonal named by `diagonal_indices[k]`. If
    `source_row_indices` is present, that diagonal is instead read from row
    `source_row_indices[k]`.
  }];

  let arguments = (ins
    AnyType:$diagonals,
    DenseI64ArrayAttr:$diagonal_indices,
    OptionalAttr<DenseI64ArrayAttr>:$source_row_indices
  );

  let results = (outs
    Kernel_PreparedLinearTransform:$prepared
  );

  let assemblyFormat = "$diagonals attr-dict `:` type($diagonals) `->` type($prepared)";
  let hasVerifier = 1;
}

def Kernel_ApplyLinearTransformOp : Kernel_Op<"apply_linear_transform", [
  Pure,
  DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop", "getOperandsToReduce"]>,
  IncreasesMulDepthOpInterface
]> {
  let summary = "Applies a prepared linear transform to a ciphertext.";
  let description = [{
    Evaluates a transform prepared by `kernel.prepare_linear_transform` on a
    ciphertext. The prepared type's parameters are verified against the
    ciphertext: applying at a level other than the one the diagonals were
    encoded at would silently compute with a wrongly-scaled transform, so a
    mismatch is a compile-time error here rather than a runtime surprise.
  }];

  let arguments = (ins
    AnyType:$input,
    Kernel_PreparedLinearTransform:$prepared
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = "$input `,` $prepared attr-dict `:` type($input) `,` type($prepared) `->` type($output)";
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_
