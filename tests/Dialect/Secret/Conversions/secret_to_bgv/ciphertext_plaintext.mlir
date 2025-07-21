// RUN: heir-opt --mlir-print-local-scope --canonicalize --secret-to-bgv %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  // CHECK: func @test_add_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_add_plain(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xi1>) -> (!eui1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xi1>
    %0 = secret.generic(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: bgv.add_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %arg1_attr : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    return %0 : !eui1
  }

  // CHECK: func @test_mul_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_mul_plain(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xi1>) -> (!eui1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xi1>
    %0 = secret.generic(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: bgv.mul_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.muli %ARG0, %arg1_attr : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    return %0 : !eui1
  }

  // CHECK: func @test_sub_plain
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_sub_plain(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xi1>) -> (!eui1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xi1>
    %0 = secret.generic(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: bgv.sub_plain %[[arg0]], %[[v0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.subi %ARG0, %arg1_attr : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    return %0 : !eui1
  }

  // CHECK: func @test_sub_plaintext_ciphertext
  // CHECK-SAME: %[[arg0:.*]]: !lwe.lwe_ciphertext
  // CHECK-SAME: %[[arg1:.*]]: tensor<1024xi1>
  func.func @test_sub_plaintext_ciphertext(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : tensor<1024xi1>) -> (!eui1 {mgmt.mgmt = #mgmt}) {
    %arg1_attr = mgmt.init %arg1 {mgmt.mgmt = #mgmt} : tensor<1024xi1>
    %0 = secret.generic(%arg0 :  !eui1) {
    // CHECK: %[[v0:.*]] = lwe.rlwe_encode %[[arg1]]
    // CHECK: bgv.sub_plain %[[v0]], %[[arg0]]
      ^bb0(%ARG0 : tensor<1024xi1>):
        %1 = arith.subi %arg1_attr, %ARG0 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    return %0 : !eui1
  }
}
