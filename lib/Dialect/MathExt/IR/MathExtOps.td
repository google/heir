#ifndef LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_TD_
#define LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_TD_

include "lib/Dialect/MathExt/IR/MathExtDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

class MathExt_Op<string mnemonic, list<Trait> traits = []> :
        Op<MathExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::math_ext";
}

def MathExt_SignOp : MathExt_Op<"sign", [Pure, ElementwiseMappable,SameOperandsAndResultType]> {
  let summary = "Returns the sign of the input value";
  let description = [{
    Returns -1 if the input is negative, 0 if it is zero, and 1 if it is positive.
    The behavior is undefined for NaN inputs.
  }];
  let arguments = (ins SignlessIntegerOrFloatLike:$value);
  let results = (outs SignlessIntegerOrFloatLike:$result);
   let assemblyFormat = "$value attr-dict `:` type($result)";
}

def MathExt_SigmoidOp : MathExt_Op<"sigmoid", [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let summary = "Returns elementwise sigmoid of the input value";
  let description = [{
    Applies the sigmoid logistic function to each element of the input:
    $$\sigma(x)=\frac{1}{1+\exp{−x}}$$
  }];
  let arguments = (ins FloatLike:$value);
  let results = (outs FloatLike:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def 1DOr2DFloatTensor : TensorRankOf<[AnyFloat], [1, 2]>;

def MathExt_SoftmaxOp : MathExt_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Returns the softmax distribution of its input vector or batch of vectors";
  let description = [{
    Applies the softmax function to the input vector or batch of vectors.
    For 2D inputs, the softmax is applied row-wise.
  }];
  let arguments = (ins 1DOr2DFloatTensor:$value);
  let results = (outs 1DOr2DFloatTensor:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

#endif  // LIB_DIALECT_MATHEXT_IR_MATHEXTOPS_TD_
