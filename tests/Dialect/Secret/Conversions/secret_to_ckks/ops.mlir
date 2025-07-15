// RUN: heir-opt --mlir-print-local-scope --canonicalize --secret-to-ckks %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>
!efi1 = !secret.secret<tensor<1024xf32>>

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>
#mgmt1 = #mgmt.mgmt<level = 0, dimension = 3>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  // CHECK: func @test_arith_ops
  func.func @test_arith_ops(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : !eui1 {mgmt.mgmt = #mgmt}, %arg2 : !eui1 {mgmt.mgmt = #mgmt}) -> (!eui1 {mgmt.mgmt = #mgmt1}) {
    %0 = secret.generic(%arg0: !eui1, %arg1: !eui1) {
    // CHECK: ckks.add
      ^bb0(%ARG0 : tensor<1024xi1>, %ARG1 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %ARG1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt})
    // CHECK: ckks.mul
    %1 = secret.generic(%0: !eui1, %arg2: !eui1) {
      ^bb0(%ARG0 : tensor<1024xi1>, %ARG1 : tensor<1024xi1>):
        %1 = arith.muli %ARG0, %ARG1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> (!eui1 {mgmt.mgmt = #mgmt1})
    // CHECK: return
    // CHECK-SAME: message_type = tensor<1024xi1>
    // CHECK-SAME: polynomialModulus = <1 + x**1024>
    // CHECK-SAME: size = 3
    return %1 : !eui1
  }

  // CHECK: func @test_arith_float_ops
  func.func @test_arith_float_ops(%arg0 : !efi1 {mgmt.mgmt = #mgmt}, %arg1 : !efi1 {mgmt.mgmt = #mgmt}, %arg2 : !efi1 {mgmt.mgmt = #mgmt}) -> (!efi1 {mgmt.mgmt = #mgmt1}) {
    %0 = secret.generic(%arg0:  !efi1, %arg1: !efi1) {
    // CHECK: ckks.add
      ^bb0(%ARG0 : tensor<1024xf32>, %ARG1 : tensor<1024xf32>):
        %1 = arith.addf %ARG0, %ARG1 : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> (!efi1 {mgmt.mgmt = #mgmt})
    // CHECK: ckks.negate
    %1 = secret.generic(%0: !efi1) {
      ^bb0(%ARG0 : tensor<1024xf32>):
        %2 = arith.negf %ARG0 : tensor<1024xf32>
        secret.yield %2 : tensor<1024xf32>
    } -> ((!efi1 {mgmt.mgmt = #mgmt}) {mgmt.mgmt = #mgmt})
    // CHECK: ckks.mul
    %3 = secret.generic(%1: !efi1, %arg2: !efi1) {
    ^bb0(%ARG0 : tensor<1024xf32>, %ARG1 : tensor<1024xf32>):
      %4 = arith.mulf %ARG0, %ARG1 : tensor<1024xf32>
      secret.yield %4 : tensor<1024xf32>
    } -> (!efi1 {mgmt.mgmt = #mgmt1})
    // CHECK: return
    // CHECK-SAME: message_type = tensor<1024xf32>
    // CHECK-SAME: polynomialModulus = <1 + x**1024>
    // CHECK-SAME: size = 3
    return %3 : !efi1
  }

  // CHECK: func @test_extract
  func.func @test_extract(%arg0 : !efi1 {mgmt.mgmt = #mgmt}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt}) {
    %0 = secret.generic(%arg0 :  !efi1) {
    // CHECK: ckks.extract
      ^bb0(%ARG0 : tensor<1024xf32>):
        %c0 = arith.constant 0 : index
        %1 = tensor.extract %ARG0[%c0] : tensor<1024xf32>
        secret.yield %1 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt})
    // CHECK: return
    // CHECK-SAME: message_type = f32
    // CHECK-SAME: polynomialModulus = <1 + x**1024>
    return %0 : !secret.secret<f32>
  }

  // Tests that a 2-D tensor is treated as a 1-D tensor along the non-unit dimension.
  // TODO(#913): Blocked on a layout representation.
  // CHECK: func @test_mul_2d
  func.func @test_mul_2d(%arg0 : !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) {
    %c0 = arith.constant dense<2.0> : tensor<1x1024xf32>
    %c0_attr = mgmt.init %c0 {mgmt.mgmt = #mgmt} : tensor<1x1024xf32>
    %0 = secret.generic(%arg0 :  !secret.secret<tensor<1x1024xf32>>) {
    // CHECK: ckks.mul_plain
      ^bb0(%ARG0 : tensor<1x1024xf32>):
        %1 = arith.mulf %ARG0, %c0_attr : tensor<1x1024xf32>
        secret.yield %1 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt})
    // CHECK: return
    // CHECK-SAME: message_type = tensor<1x1024xf32>
    // CHECK-SAME: polynomialModulus = <1 + x**1024>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }

  // CHECK: func.func private @callee_secret
  // CHECK: func @test_call
  func.func private @callee(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @test_call(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) {
    // CHECK: call @callee_secret
    %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = func.call @callee(%input0) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %1 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt})
    // CHECK: return
    // CHECK-SAME: message_type = tensor<1x1024xf32>
    // CHECK-SAME: coefficientType = !rns.rns<!mod_arith.int<36028797019389953 : i64>>, polynomialModulus = <1 + x**1024>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
