#ifndef LIB_TRANSFORMS_CONVERTSECRETINSERTTOSTATICINSERT_CONVERTSECRETINSERTTOSTATICINSERT_TD_
#define LIB_TRANSFORMS_CONVERTSECRETINSERTTOSTATICINSERT_CONVERTSECRETINSERTTOSTATICINSERT_TD_

include "mlir/Pass/PassBase.td"

def ConvertSecretInsertToStaticInsert : Pass<"convert-secret-insert-to-static-insert"> {
  let summary = "Convert `tensor.insert` operations on secret index to static insert operations.";
  let description = [{
  Converts `tensor.insert` operations that write to secret index to alternative static `tensor.insert` operations that inserts the inserted value at each index and conditionally selects the newly produced tensor that contains the value at the secret index.

  Note: Running this pass alone does not result in a data-oblivious program; we have to run the `--convert-if-to-select` pass to the resulting program to convert the secret-dependent If-operation to a Select-operation.

  Example input:
    ```mlir
    func.func @main(%secretTensor: !secret.secret<tensor<32xi16>>, %secretIndex: !secret.secret<index>)) -> !secret.secret<i16> {
      ...
      %0 = secret.generic ins(%secretTensor, %secretIndex : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
      ^bb0(%tensor: tensor<32xi16>, %index: index):
        // Violation: tensor.insert writes value at secret index
        %inserted = tensor.insert %newValue into %tensor[%index] : tensor<16xi32>
      ...
    }
    ```


    Output:
    ```mlir
    func.func @main(%secretTensor: !secret.secret<tensor<32xi16>>, %secretIndex: !secret.secret<index>)) -> !secret.secret<i16> {
      ...
      %0 = secret.generic ins(%secretTensor, %secretIndex : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
      ^bb0(%tensor: tensor<32xi16>, %index: index):
        %inserted = affine.for %i=0 to 16 iter_args(%inputArg = %tensor) -> tensor<16xi32> {
          // 1. Check if %i matches the %index
          %cond = arith.cmpi eq, %i, %index : index
          // 2. Insert %newValue and produce %newTensor
          %newTensor = tensor.insert %value into %inputArg[%i] : tensor<16xi32>
          // 3. If %i matches %inputIndex, yield %newTensor, else yield unchanged input tensor
          %finalTensor = scf.if %cond -> (i32) {
            scf.yield %newTensor : tensor<16xi32>
          } else{
            scf.yield %inputArg : tensor<16xi32>
          }
          // 4. Yield final tensor
          affine.yield %finalTensor : tensor<16xi32>
  }
      ...
    }
    ```

  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTSECRETINSERTTOSTATICINSERT_CONVERTSECRETINSERTTOSTATICINSERT_TD_
