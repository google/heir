module {
  func.func @basic_test(%arg0: tensor<8192xi32>, %arg1: tensor<8192xi32>, %arg2: tensor<8192xi32>, %arg3: tensor<8192xi32>) -> (tensor<8192xi32>, tensor<8192xi32>) {
    %0 = pisa.add %arg0, %arg2 {i = 0 : i32, q = 463187969 : i32} : tensor<8192xi32>
    %1 = pisa.add %arg1, %arg3 {i = 0 : i32, q = 463187969 : i32} : tensor<8192xi32>
    return %0, %1 : tensor<8192xi32>, tensor<8192xi32>
  }
}
