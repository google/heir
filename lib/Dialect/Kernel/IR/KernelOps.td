#ifndef LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_
#define LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_

include "KernelDialect.td"

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
  DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop", "getOperandToReduce"]>,
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
  DeclareOpInterfaceMethods<ReducesLevelOpInterface, ["getLevelsToDrop", "getOperandToReduce"]>,
  ElementwiseMappable,
  IncreasesMulDepthOpInterface
]> {
  let summary = "Performs a linear transform (matrix-vector multiplication) on a ciphertext.";
  let description = [{
    Performs a linear transform on a ciphertext using diagonal method.
  }];

  let arguments = (ins
    AnyType:$input,
    ElementsAttr:$diagonals,
    DenseI64ArrayAttr:$diagonal_indices,
    OptionalAttr<F64Attr>:$bsgs_ratio
  );

  let results = (outs
    AnyType:$output
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

#endif  // LIB_DIALECT_KERNEL_IR_KERNELOPS_TD_
