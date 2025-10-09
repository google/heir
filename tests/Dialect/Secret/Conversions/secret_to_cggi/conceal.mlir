// RUN: heir-opt --secret-distribute-generic --secret-to-cggi --split-input-file %s | FileCheck %s

// CHECK: ![[ct_ty:.*]] = !lwe.lwe_ciphertext

// CHECK: @conceal
// CHECK-SAME: ([[ARG:%.*]]: i3) -> tensor<3x![[ct_ty]]>
// CHECK:    %[[c1_i3:.*]] = arith.constant 1 : i3
// CHECK:    %[[v0:.*]] = arith.andi [[ARG]], %[[c1_i3]] : i3
// CHECK:    %[[v1:.*]] = arith.trunci %[[v0]] : i3 to i1
// CHECK:    %[[pt:.*]] = lwe.encode %[[v1]]
// CHECK:    %[[ct:.*]] = lwe.trivial_encrypt %[[pt]]
// CHECK-COUNT-2: lwe.trivial_encrypt
// CHECK-NOT: lwe.trivial_encrypt
// CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ct]], %[[ct_1:.*]], %[[ct_2:.*]]
// CHECK: return %[[TENSOR]] : tensor<3x![[ct_ty]]>
func.func @conceal(%arg0: i3) -> !secret.secret<i3> {
  %0 = secret.conceal %arg0 : i3 -> !secret.secret<i3>
  func.return %0 : !secret.secret<i3>
}

// -----

// CHECK: @conceal_memref
// CHECK-SAME: ([[ARG:%.*]]: tensor<3xi1>) -> tensor<3x![[ct_ty]]>
func.func @conceal_memref(%arg0: tensor<3xi1>) -> !secret.secret<tensor<3xi1>> {
  // CHECK-COUNT-3: tensor.extract
  // CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ct:.*]], %[[ct_1:.*]], %[[ct_2:.*]]
  // CHECK: return %[[TENSOR]] : tensor<3x![[ct_ty]]>
  %0 = secret.conceal %arg0 : tensor<3xi1> -> !secret.secret<tensor<3xi1>>
  func.return %0 : !secret.secret<tensor<3xi1>>
}

// -----

// CHECK: @conceal_memref_iN
// CHECK-SAME: ([[ARG:%.*]]: tensor<3xi2>) -> tensor<3x2x![[ct_ty]]>
func.func @conceal_memref_iN(%arg0: tensor<3xi2>) -> !secret.secret<tensor<3xi2>> {
  // CHECK-COUNT-3: tensor.extract
  // CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[ct:.*]], %[[ct_1:.*]], %[[ct_2:.*]]
  // CHECK: return %[[TENSOR]] : tensor<3x2x![[ct_ty]]>
  %0 = secret.conceal %arg0 : tensor<3xi2> -> !secret.secret<tensor<3xi2>>
  func.return %0 : !secret.secret<tensor<3xi2>>
}

// -----

// CHECK: @conceal_i1
// CHECK-SAME: ([[ARG:%.*]]: i1) -> [[TY:!.*]]
func.func @conceal_i1(%arg0: i1) -> !secret.secret<i1> {
  // CHECK: lwe.encode
  // CHECK-NEXT: lwe.trivial_encrypt
  // CHECK-NEXT: return
  %0 = secret.conceal %arg0 : i1 -> !secret.secret<i1>
  func.return %0 : !secret.secret<i1>
}

// -----

// CHECK: @conceal_MxN
// CHECK-SAME: ([[ARG:%.*]]: tensor<3x3xi1>) -> tensor<3x3x![[ct_ty]]>
func.func @conceal_MxN(%arg0: tensor<3x3xi1>) -> !secret.secret<tensor<3x3xi1>> {
  // CHECK-COUNT-9: tensor.extract
  // CHECK: return
  %0 = secret.conceal %arg0 : tensor<3x3xi1> -> !secret.secret<tensor<3x3xi1>>
  func.return %0 : !secret.secret<tensor<3x3xi1>>
}
