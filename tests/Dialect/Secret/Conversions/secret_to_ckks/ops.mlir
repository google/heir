// RUN: heir-opt --mlir-print-local-scope --canonicalize --secret-to-ckks %s | FileCheck %s

!eui1 = !secret.secret<tensor<1024xi1>>
!efi1 = !secret.secret<tensor<1024xf32>>

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>
#mgmt1 = #mgmt.mgmt<level = 0, dimension = 3>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  // CHECK-LABEL: func @test_arith_ops
  func.func @test_arith_ops(%arg0 : !eui1 {mgmt.mgmt = #mgmt}, %arg1 : !eui1 {mgmt.mgmt = #mgmt}, %arg2 : !eui1 {mgmt.mgmt = #mgmt}) -> (!eui1 {mgmt.mgmt = #mgmt1}) {
    %0 = secret.generic ins(%arg0, %arg1 :  !eui1, !eui1) attrs = {__resattrs = [{mgmt.mgmt = #mgmt}]} {
    // CHECK: ckks.add
      ^bb0(%ARG0 : tensor<1024xi1>, %ARG1 : tensor<1024xi1>):
        %1 = arith.addi %ARG0, %ARG1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    // CHECK: ckks.mul
    %1 = secret.generic ins(%0, %arg2 :  !eui1, !eui1) attrs = {__resattrs = [{mgmt.mgmt = #mgmt1}]} {
      ^bb0(%ARG0 : tensor<1024xi1>, %ARG1 : tensor<1024xi1>):
        %1 = arith.muli %ARG0, %ARG1 : tensor<1024xi1>
        secret.yield %1 : tensor<1024xi1>
    } -> !eui1
    // CHECK: return
    // CHECK-SAME: message_type = tensor<1024xi1>
    // CHECK-SAME: polynomialModulus = <1 + x**1024>
    // CHECK-SAME: size = 3
    return %1 : !eui1
  }

  // CHECK-LABEL: func @test_arith_float_ops
  func.func @test_arith_float_ops(%arg0 : !efi1 {mgmt.mgmt = #mgmt}, %arg1 : !efi1 {mgmt.mgmt = #mgmt}, %arg2 : !efi1 {mgmt.mgmt = #mgmt}) -> (!efi1 {mgmt.mgmt = #mgmt1}) {
    %0 = secret.generic ins(%arg0, %arg1 :  !efi1, !efi1) attrs = {__resattrs = [{mgmt.mgmt = #mgmt}]} {
    // CHECK: ckks.add
      ^bb0(%ARG0 : tensor<1024xf32>, %ARG1 : tensor<1024xf32>):
        %1 = arith.addf %ARG0, %ARG1 : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !efi1
    // CHECK: ckks.mul
    %1 = secret.generic ins(%0, %arg2 :  !efi1, !efi1) attrs = {__resattrs = [{mgmt.mgmt = #mgmt1}]} {
      ^bb0(%ARG0 : tensor<1024xf32>, %ARG1 : tensor<1024xf32>):
        %1 = arith.mulf %ARG0, %ARG1 : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !efi1
    // CHECK: return
    // CHECK-SAME: message_type = tensor<1024xf32>
    // CHECK-SAME: polynomialModulus = <1 + x**1024>
    // CHECK-SAME: size = 3
    return %1 : !efi1
  }

  // CHECK-LABEL: func @test_extract
  func.func @test_extract(%arg0 : !efi1 {mgmt.mgmt = #mgmt}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt}) {
    %0 = secret.generic ins(%arg0 :  !efi1) attrs = {__resattrs = [{mgmt.mgmt = #mgmt}]} {
    // CHECK: ckks.extract
      ^bb0(%ARG0 : tensor<1024xf32>):
        %c0 = arith.constant 0 : index
        %1 = tensor.extract %ARG0[%c0] : tensor<1024xf32>
        secret.yield %1 : f32
    } -> !secret.secret<f32>
    // CHECK: return
    // CHECK-SAME: message_type = f32
    // CHECK-SAME: polynomialModulus = <1 + x**1024>
    return %0 : !secret.secret<f32>
  }

  // Tests that a 2-D tensor is treated as a 1-D tensor along the non-unit dimension.
  // TODO(#913): Blocked on a layout representation.
  // CHECK-LABEL: func @test_mul_2d
  func.func @test_mul_2d(%arg0 : !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) {
    %0 = secret.generic ins(%arg0 :  !secret.secret<tensor<1x1024xf32>>) attrs = {__resattrs = [{mgmt.mgmt = #mgmt}]} {
    // CHECK: ckks.mul_plain
      ^bb0(%ARG0 : tensor<1x1024xf32>):
        %c0 = arith.constant dense<2.0> : tensor<1x1024xf32>
        %1 = arith.mulf %ARG0, %c0: tensor<1x1024xf32>
        secret.yield %1 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK: return
    // CHECK-SAME: message_type = tensor<1x1024xf32>
    // CHECK-SAME: polynomialModulus = <1 + x**1024>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }

  // CHECK-LABEL: func.func private @callee_secret
  // CHECK-LABEL: func @test_call
  func.func private @callee(tensor<1x1024xf32>) -> tensor<1x1024xf32>
  func.func @test_call(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) {
    // CHECK: call @callee_secret
    %0 = secret.generic ins(%arg0 : !secret.secret<tensor<1x1024xf32>>) attrs = {__resattrs = [{mgmt.mgmt = #mgmt}]} {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = func.call @callee(%input0) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      secret.yield %1 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    // CHECK: return
    // CHECK-SAME: message_type = tensor<1x1024xf32>
    // CHECK-SAME: coefficientType = !rns.rns<!mod_arith.int<36028797019389953 : i64>>, polynomialModulus = <1 + x**1024>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
