#ifndef LIB_TRANSFORMS_CONVERTSECRETEXTRACTTOSTATICEXTRACT_CONVERTSECRETEXTRACTTOSTATICEXTRACT_TD_
#define LIB_TRANSFORMS_CONVERTSECRETEXTRACTTOSTATICEXTRACT_CONVERTSECRETEXTRACTTOSTATICEXTRACT_TD_

include "mlir/Pass/PassBase.td"

def ConvertSecretExtractToStaticExtract : Pass<"convert-secret-extract-to-static-extract"> {
  let summary = "Convert `tensor.extract` operations on secret index to static extract operations.";
  let description = [{Converts `tensor.extract` operations that read value at secret index to alternative static `tensor.extract` operations that extracts value at each index and conditionally selects the value extracted at the secret index.

  Note: Running this pass alone does not result in a data-oblivious program; we have to run the `--convert-if-to-select` pass to the resulting program to convert the secret-dependent If-operation to a Select-operation.

  Example input:
    ```mlir
    func.func @main(%secretTensor: !secret.secret<tensor<32xi16>>, %secretIndex: !secret.secret<index>)) -> !secret.secret<i16> {
      ...
      %0 = secret.generic ins(%secretTensor, %secretIndex : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
      ^bb0(%tensor: tensor<32xi16>, %index: index):
        // Violation: tensor.extract loads value at secret index
        %extractedValue = tensor.extract %tensor[%index] : tensor<16xi32>
      ...
    }
    ```


    Output:
    ```mlir
    func.func @main(%secretTensor: !secret.secret<tensor<32xi16>>, %secretIndex: !secret.secret<index>)) -> !secret.secret<i16> {
      ...
      %0 = secret.generic ins(%secretTensor, %secretIndex : !secret.secret<tensor<32xi16>>, !secret.secret<index>) {
      ^bb0(%tensor: tensor<32xi16>, %index: index):
        %extractedValue = affine.for %i=0 to 16 iter_args(%arg= %dummyValue) -> (i32) {
          // 1. Check if %i matches %index
          %cond = arith.cmpi eq, %i, %index : index
          // 2. Extract value at %i
          %value = tensor.extract %tensor[%i] : tensor<16xi32>
          // 3. If %i matches %index, yield %value extracted in (2), else yield %dummyValue
          %result = scf.if %cond -> (i32) {
            scf.yield %value : i32
          } else{
            scf.yield %arg : i32
          }
          // 4. Yield result from (3)
          affine.yield %result : i32
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

#endif  // LIB_TRANSFORMS_CONVERTSECRETEXTRACTTOSTATICEXTRACT_CONVERTSECRETEXTRACTTOSTATICEXTRACT_TD_
