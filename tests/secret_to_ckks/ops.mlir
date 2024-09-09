// RUN: heir-opt --canonicalize --secret-to-ckks %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>
!efi1 = !secret.secret<tensor<1024xf32>>

module {
  // CHECK-LABEL: func @test_arith_ops
  func.func @test_arith_ops(%arg0 : !eui1, %arg1 : !eui1, %arg2 : !eui1) -> (!eui1) {
    %0 = secret.generic ins(%arg0, %arg1 :  !eui1, !eui1) {
    // CHECK: ckks.add
      ^bb0(%ARG0 : tensor<1024xi1>, %ARG1 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %ARG1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    // CHECK: ckks.mul
    // CHECK-NEXT: ckks.relinearize
    %1 = secret.generic ins(%0, %arg2 :  !eui1, !eui1) {
      ^bb0(%ARG0 : tensor<1024xi1>, %ARG1 : tensor<1024xi1>):
        %1 = arith.muli %ARG0, %ARG1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    // CHECK: return
    // CHECK-SAME: coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus = <1 + x**1024>
    return %1 : !eui1
  }

  // CHECK-LABEL: func @test_arith_float_ops
  func.func @test_arith_float_ops(%arg0 : !efi1, %arg1 : !efi1, %arg2 : !efi1) -> (!efi1) {
    %0 = secret.generic ins(%arg0, %arg1 :  !efi1, !efi1) {
    // CHECK: ckks.add
      ^bb0(%ARG0 : tensor<1024xf32>, %ARG1 : tensor<1024xf32>):
        %1 = arith.addf %ARG0, %ARG1 : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !efi1
    // CHECK: ckks.mul
    // CHECK-NEXT: ckks.relinearize
    %1 = secret.generic ins(%0, %arg2 :  !efi1, !efi1) {
      ^bb0(%ARG0 : tensor<1024xf32>, %ARG1 : tensor<1024xf32>):
        %1 = arith.mulf %ARG0, %ARG1 : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !efi1
    // CHECK: return
    // CHECK-SAME: coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus = <1 + x**1024>
    return %1 : !efi1
  }
}
