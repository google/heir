// RUN: heir-opt --canonicalize --secret-to-ckks %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>
!efi1 = !secret.secret<tensor<1024xf32>>

module {
  // CHECK-LABEL: func @test_addi_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_addi_plain(%arg0 : !eui1, %arg1 : tensor<1024xi1>) -> (!eui1) {
    %0 = secret.generic ins(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.add_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %arg1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    return %0 : !eui1
  }

  // CHECK-LABEL: func @test_muli_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_muli_plain(%arg0 : !eui1, %arg1 : tensor<1024xi1>) -> (!eui1) {
    %0 = secret.generic ins(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.mul_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.muli %ARG0, %arg1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    return %0 : !eui1
  }

  // CHECK-LABEL: func @test_subi_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_subi_plain(%arg0 : !eui1, %arg1 : tensor<1024xi1>) -> (!eui1) {
    %0 = secret.generic ins(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.sub_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.subi %ARG0, %arg1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    return %0 : !eui1
  }

  // CHECK-LABEL: func @test_addf_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xf32>
  func.func @test_addf_plain(%arg0 : !efi1, %arg1 : tensor<1024xf32>) -> (!efi1) {
    %0 = secret.generic ins(%arg0 :  !efi1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.add_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xf32>):
        %1 = arith.addf %ARG0, %arg1 : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !efi1
    return %0 : !efi1
  }

  // CHECK-LABEL: func @test_mulf_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xf32>
  func.func @test_mulf_plain(%arg0 : !efi1, %arg1 : tensor<1024xf32>) -> (!efi1) {
    %0 = secret.generic ins(%arg0 :  !efi1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.mul_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xf32>):
        %1 = arith.mulf %ARG0, %arg1 : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !efi1
    return %0 : !efi1
  }

  // CHECK-LABEL: func @test_subf_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.rlwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xf32>
  func.func @test_subf_plain(%arg0 : !efi1, %arg1 : tensor<1024xf32>) -> (!efi1) {
    %0 = secret.generic ins(%arg0 :  !efi1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.sub_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xf32>):
        %1 = arith.subf %ARG0, %arg1 : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !efi1
    return %0 : !efi1
  }
}
