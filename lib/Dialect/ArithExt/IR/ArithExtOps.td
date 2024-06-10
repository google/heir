#ifndef LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_TD_
#define LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_TD_

include "lib/Dialect/ArithExt/IR/ArithExtDialect.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class ArithExt_Op<string mnemonic, list<Trait> traits = [Pure]> :
        Op<ArithExt_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::arith_ext";
}

def ArithExt_BarrettReduceOp : ArithExt_Op<"barrett_reduce", [SameOperandsAndResultType]> {
  let summary = "Compute the first step of the Barrett reduction.";
  let description = [{
    Let $q$ denote a statically known modulus and $b = 4^{w}$, where $w$ is the
    smallest bit-width that contains the range $[0, q)$. The Barrett reduce
    operation computes `barret_reduce x = x - floor(x * floor(b / q) / b) * q`.

    Given $0 <= x < q^2$, then this will compute $(x \mod q)$ or $(x \mod q) + p$.
  }];

  let arguments = (ins
    SignlessIntegerLike:$input,
    I64Attr:$modulus
  );
  let results = (outs SignlessIntegerLike:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($input))";

  let hasVerifier = 1;
}

def ArithExt_SubIfGEOp : ArithExt_Op<"subifge", [SameOperandsAndResultType]> {
  let summary = "Compute (x >= y) ? x - y : x.";

  let arguments = (ins
    SignlessIntegerLike:$lhs,
    SignlessIntegerLike:$rhs
  );
  let results = (outs SignlessIntegerLike:$output);
  let assemblyFormat = "operands attr-dict `:` qualified(type($output))";
}

#endif  // LIB_DIALECT_ARITHEXT_IR_ARITHEXTOPS_TD_
