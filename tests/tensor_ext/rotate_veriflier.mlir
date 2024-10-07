// RUN: heir-opt --verify-diagnostics --split-input-file %s


func.func @test_rotate_verifier(%0: tensor<10x16xi32>) {
  %c1 = arith.constant 1 : i32
  // expected-error@+1 {{requires a 1-D input tensor or tensor with single non-unit dimension}}
  %1 = tensor_ext.rotate %0, %c1 : tensor<10x16xi32>, i32
  return
}

// -----

func.func @test_rotate_verifier_ok(%0: tensor<1x16xi32>) {
  %c1 = arith.constant 1 : i32
  %1 = tensor_ext.rotate %0, %c1 : tensor<1x16xi32>, i32
  return
}
