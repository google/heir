#ifndef LIB_TRANSFORMS_CONVERTSECRETFORTOSTATICFOR_CONVERTSECRETFORTOSTATICFOR_TD_
#define LIB_TRANSFORMS_CONVERTSECRETFORTOSTATICFOR_CONVERTSECRETFORTOSTATICFOR_TD_

include "mlir/Pass/PassBase.td"

def ConvertSecretForToStaticFor : Pass<"convert-secret-for-to-static-for"> {
  let summary = "Convert secret scf.for ops to affine.for ops with constant bounds.";
  let description = [{
  Conversion for For-operation that evaluate secret bound(s) to alternative affine For-operation with constant bound(s).

  It replaces data-dependent bounds with an If-operation to check the bounds, and conditionally execute and yield values from the For-operation's body.
  Note: Running this pass alone does not result in a data-oblivious program; we have to run the `--convert-if-to-select` pass to the resulting program to convert the secret-dependent If-operation to a Select-operation.

  Example input:
    ```mlir
      func.func @main(%secretTensor: !secret.secret<tensor<16xi32>>, %secretLower: !secret.secret<index>, %secretUpper: !secret.secret<index>) -> !secret.secret<i32> {
       ...
       %0 = secret.generic ins(%secretTensor, %secretLower, %secretUpper : !secret.secret<tensor<16xi32>>, !secret.secret<index>, !secret.secret<index>){
        ^bb0(%tensor: tensor<16xi32>, %lower : index, %upper : index ):
          ...
          %1 = scf.for %i = %lower to %upper step %step iter_args(%arg = %val) -> (i32) {
            %extracted = tensor.extract %input[%i] : tensor<16xi32>
            %sum = arith.addi %extracted, %arg : i32
            scf.yield %sum : i32
          } {lower = 0, upper = 16}
          secret.yield %1 : i32
      } -> !secret.secret<i32>
      return %0 : !secret.secret<i32>
    ```

    Output:
    ```mlir
      func.func @main(%secretTensor: !secret.secret<tensor<16xi32>>, %secretIndex: !secret.secret<index> {secret.secret}) -> !secret.secret<i32> {
       ...
       %0 = secret.generic ins(%secretTensor, %secretLower, %secretUpper : !secret.secret<tensor<16xi32>>, !secret.secret<index>, !secret.secret<index>){
        ^bb0(%tensor: tensor<16xi32>, %lower : index, %upper : index ):
          ...
          %1 = affine.for %i = 0 to 16 step %step iter_args(%arg = %val) -> (i32) {
            %lowerCond = arith.cmpi sge, %i, %index : index
            %upperCond = arith.cmpi slt, %i, %index : index
            %cond = arith.andi %lowerCond, %upperCond : i1
            %result = scf.if(%cond) -> (i32) {
              %extracted = tensor.extract %input[%i] : tensor<16xi32>
              %sum = arith.addi %extracted, %arg : i32
              scf.yield %sum : i32
            } else {
              scf.yield %arg : i32
            }
            affine.yield %result : i32
          } {lower = 0, upper = 16}
          secret.yield %1 : i32
      } -> !secret.secret<i32>
      return %0 : !secret.secret<i32>
    ```
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTSECRETFORTOSTATICFOR_CONVERTSECRETFORTOSTATICFOR_TD_
