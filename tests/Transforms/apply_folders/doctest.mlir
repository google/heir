// RUN: heir-opt --apply-folders %s

func.func @full_inserts(%arg0: i32, %arg1: i32, %arg2: i32) -> (tensor<3xi32>) {
  %cst = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %inserted = tensor.insert %arg0 into %cst[%c0] : tensor<3xi32>
  %inserted_1 = tensor.insert %arg1 into %inserted[%c1] : tensor<3xi32>
  %inserted_2 = tensor.insert %arg2 into %inserted_1[%c2] : tensor<3xi32>
  return %inserted_2 : tensor<3xi32>
}
