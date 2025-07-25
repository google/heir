// RUN: heir-opt --mlir-print-local-scope --canonicalize --secret-to-ckks %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>
!efi1 = !secret.secret<tensor<1024xf32>>

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  // CHECK: func @test_addi_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_addi_plain(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xi1>) -> (!eui1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xi1>
    %0 = secret.generic(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.add_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %arg1_attr : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    return %0 : !eui1
  }

  // CHECK: func @test_muli_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_muli_plain(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xi1>) -> (!eui1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xi1>
    %0 = secret.generic(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.mul_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.muli %ARG0, %arg1_attr : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    return %0 : !eui1
  }

  // CHECK: func @test_subi_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_subi_plain(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xi1>) -> (!eui1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xi1>
    %0 = secret.generic(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.sub_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.subi %ARG0, %arg1_attr : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    return %0 : !eui1
  }

  // CHECK: func @test_addf_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xf32>
  func.func @test_addf_plain(%arg0 : !efi1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xf32>) -> (!efi1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xf32>
    %0 = secret.generic(%arg0 :  !efi1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.add_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xf32>):
        %1 = arith.addf %ARG0, %arg1_attr : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> (!efi1 {mgmt.mgmt = #mgmt})
    return %0 : !efi1
  }

  // CHECK: func @test_mulf_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xf32>
  func.func @test_mulf_plain(%arg0 : !efi1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xf32>) -> (!efi1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xf32>
    %0 = secret.generic(%arg0 :  !efi1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.mul_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xf32>):
        %1 = arith.mulf %ARG0, %arg1_attr : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> (!efi1 {mgmt.mgmt = #mgmt})
    return %0 : !efi1
  }

  // CHECK: func @test_subf_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xf32>
  func.func @test_subf_plain(%arg0 : !efi1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xf32>) -> (!efi1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xf32>
    %0 = secret.generic(%arg0 :  !efi1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: ckks.sub_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xf32>):
        %1 = arith.subf %ARG0, %arg1_attr : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> (!efi1 {mgmt.mgmt = #mgmt})
    return %0 : !efi1
  }
}
