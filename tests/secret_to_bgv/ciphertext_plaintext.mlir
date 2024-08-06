// RUN: heir-opt --canonicalize --secret-to-bgv %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>

module {
  // CHECK-LABEL: func @test_add_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_add_plain(%arg0 : !eui1, %arg1 : tensor<1024xi1>) -> (!eui1) {
    %0 = secret.generic ins(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: bgv.add_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %arg1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    return %0 : !eui1
  }

  // CHECK-LABEL: func @test_mul_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_mul_plain(%arg0 : !eui1, %arg1 : tensor<1024xi1>) -> (!eui1) {
    %0 = secret.generic ins(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: bgv.mul_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.muli %ARG0, %arg1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    return %0 : !eui1
  }

  // CHECK-LABEL: func @test_sub_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_sub_plain(%arg0 : !eui1, %arg1 : tensor<1024xi1>) -> (!eui1) {
    %0 = secret.generic ins(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: bgv.sub_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.subi %ARG0, %arg1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    return %0 : !eui1
  }
}
