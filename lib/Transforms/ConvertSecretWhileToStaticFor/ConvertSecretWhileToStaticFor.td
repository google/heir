#ifndef LIB_TRANSFORMS_CONVERTSECRETWHILETOSTATICFOR_CONVERTSECRETWHILETOSTATICFOR_TD_
#define LIB_TRANSFORMS_CONVERTSECRETWHILETOSTATICFOR_CONVERTSECRETWHILETOSTATICFOR_TD_

include "mlir/Pass/PassBase.td"

def ConvertSecretWhileToStaticFor : Pass<"convert-secret-while-to-static-for"> {
  let summary = "Convert secret scf.while ops to affine.for ops that have constant bounds.";
  let description = [{
  Convert scf.while with a secret condition to affine.for with constant bounds. It replaces the scf.condition operation found in the scf.while loop with an scf.if operation that conditionally executes operations in the while operation's body and yields values.

  A "max_iter" attribute should be specified as part of the secret-dependent scf.while operation to successfully transform to a secret-independent affine.for operation. This attribute determines the maximum number of iterations for the new affine.for operation.

  Note: Running this pass alone does not result in a data-oblivious program; we have to run the `--convert-if-to-select` pass to the resulting program to convert the secret-dependent If-operation to a Select-operation.

  Example input:

    ```mlir
    // C-like code
    int main(int secretInput) {
      while (secretInput > 100) {
        secretInput = secretInput * secretInput;
      }
      return secretInput;
    }

    // MLIR
    func.func @main(%secretInput: !secret.secret<i16>) -> !secret.secret<i16> {
      %c100 = arith.constant 100 : i16
      %0 = secret.generic ins(%secretInput : !secret.secret<i16>) {
      ^bb0(%input: i16):
        %1 = scf.while (%arg1 = %input) : (i16) -> i16 {
          %2 = arith.cmpi sgt, %arg1, %c100 : i16
          scf.condition(%2) %arg1 : i16
        } do {
        ^bb0(%arg1: i16):
          %3 = arith.muli %arg1, %arg1 : i16
          scf.yield %3 : i16
        } attributes {max_iter = 16 : i64}
        secret.yield %1 : i16
      } -> !secret.secret<i16>
      return %0 : !secret.secret<i16>
    }

    ```
    Output:
    ```mlir
    func.func @main(%secretInput: !secret.secret<i16>) -> !secret.secret<i16> {
      %c100 = arith.constant 100 : i16
      %0 = secret.generic ins(%secretInput : !secret.secret<i16>) {
      ^bb0(%input: i16):
        %1 = affine.for 0 to 16 iter_args(%arg1 = %input) -> (i16) {
          %2 = arith.cmpi sgt, %arg1, %c100 : i16
          %3 = scf.if (%2) -> i16{
            %4 = arith.muli %arg1, %arg1 : i16
            scf.yield %4 : i16
          } else {
            scf.yield %arg1 : i16
          }
          affine.yield %3 : i16
        } attributes {max_iter = 16 : i64}
        secret.yield %1 : i16
      } -> !secret.secret<i16>
      return %0 : !secret.secret<i16>
    }
    ```
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTSECRETWHILETOSTATICFOR_CONVERTSECRETWHILETOSTATICFOR_TD_
