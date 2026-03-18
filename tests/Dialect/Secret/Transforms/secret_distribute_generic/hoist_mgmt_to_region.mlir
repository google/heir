// RUN: heir-opt --secret-distribute-generic --verify-diagnostics %s | FileCheck %s

#mgmt = #mgmt.mgmt<level = 0, scale = 45>
#mgmt1 = #mgmt.mgmt<level = 1, scale = 45>

// CHECK: @test_hoist_if
func.func @test_hoist_if(%arg0: !secret.secret<tensor<1x1024xf32>>, %cond: i1) -> !secret.secret<tensor<1x1024xf32>> {
  // CHECK: scf.if
  // CHECK-NEXT: secret.generic
  // CHECK-NEXT: ^body
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: secret.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT: secret.generic
  // CHECK-NEXT: ^body
  // CHECK-NEXT: arith.subf
  // CHECK-NEXT: secret.yield
  // CHECK-NEXT: }
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: } {mgmt.mgmt =
  %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) {
  ^body(%input0: tensor<1x1024xf32>):
    %1 = scf.if %cond -> (tensor<1x1024xf32>) {
      %res = arith.addf %input0, %input0 {mgmt.mgmt = #mgmt} : tensor<1x1024xf32>
      scf.yield %res : tensor<1x1024xf32>
    } else {
      %res = arith.subf %input0, %input0 {mgmt.mgmt = #mgmt} : tensor<1x1024xf32>
      scf.yield %res : tensor<1x1024xf32>
    }
    secret.yield %1 : tensor<1x1024xf32>
  } -> !secret.secret<tensor<1x1024xf32>>
  return %0 : !secret.secret<tensor<1x1024xf32>>
}

// CHECK: @test_hoist_for
func.func @test_hoist_for(%arg0: !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // CHECK: scf.for
  // CHECK-NEXT: secret.generic
  // CHECK-NEXT: ^body
  // CHECK-NEXT: arith.addf
  // CHECK-NEXT: secret.yield
  // CHECK-NEXT: {mgmt.mgmt =
  // CHECK-NEXT: scf.yield
  // CHECK-NEXT: __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>}]
  %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) {
  ^body(%input0: tensor<1x1024xf32>):
    %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg1 = %input0) -> (tensor<1x1024xf32>) {
      %res = arith.addf %arg1, %arg1 {mgmt.mgmt = #mgmt} : tensor<1x1024xf32>
      scf.yield %res : tensor<1x1024xf32>
    } {__argattrs = [{}, {}, {}, {mgmt.mgmt = #mgmt}], __resattrs = [{mgmt.mgmt = #mgmt}]}
    secret.yield %1 : tensor<1x1024xf32>
  } -> !secret.secret<tensor<1x1024xf32>>
  return %0 : !secret.secret<tensor<1x1024xf32>>
}

// CHECK: @test_hoist_if_in_for
func.func @test_hoist_if_in_for(%arg0: !secret.secret<tensor<1x1024xf32>>, %cond: i1) -> !secret.secret<tensor<1x1024xf32>> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // CHECK: scf.for
  // CHECK: scf.if
  // CHECK: } {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>}
  // CHECK: scf.yield
  // CHECK: } {__argattrs = {{.*}}, __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>}]}
  %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt}) {
  ^body(%input0: tensor<1x1024xf32>):
    %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg1 = %input0) -> (tensor<1x1024xf32>) {
      %2 = scf.if %cond -> (tensor<1x1024xf32>) {
        %res = arith.addf %arg1, %arg1 {mgmt.mgmt = #mgmt} : tensor<1x1024xf32>
        scf.yield %res : tensor<1x1024xf32>
      } else {
        %res = arith.subf %arg1, %arg1 {mgmt.mgmt = #mgmt} : tensor<1x1024xf32>
        scf.yield %res : tensor<1x1024xf32>
      }
      scf.yield %2 : tensor<1x1024xf32>
    } {__argattrs = [{}, {}, {}, {mgmt.mgmt = #mgmt}], __resattrs = [{mgmt.mgmt = #mgmt}]}
    secret.yield %1 : tensor<1x1024xf32>
  } -> !secret.secret<tensor<1x1024xf32>>
  return %0 : !secret.secret<tensor<1x1024xf32>>
}

// CHECK: @test_hoist_inconsistent
func.func @test_hoist_inconsistent(%arg0: tensor<1x1024xf32>, %cond: i1) -> tensor<1x1024xf32> {
  // expected-warning @+1 {{Inconsistent mgmt attributes for result 0}}
  %1 = scf.if %cond -> (tensor<1x1024xf32>) {
    %res = arith.addf %arg0, %arg0 {mgmt.mgmt = #mgmt} : tensor<1x1024xf32>
    scf.yield %res : tensor<1x1024xf32>
  } else {
    %res = arith.subf %arg0, %arg0 {mgmt.mgmt = #mgmt1} : tensor<1x1024xf32>
    scf.yield %res : tensor<1x1024xf32>
  }
  return %1 : tensor<1x1024xf32>
}
