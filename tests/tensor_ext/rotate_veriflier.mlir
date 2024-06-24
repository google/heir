// RUN: heir-opt --verify-diagnostics %s


func.func @test_rotate_verifier(%0: tensor<10x16xi32>) {
  %c1 = arith.constant 1 : i32
  // expected-error@+1 {{requires a 1-D input tensor, but found 'tensor<10x16xi32>'}}
  %1 = tensor_ext.rotate %0, %c1 : tensor<10x16xi32>, i32
  return
}
