// RUN: heir-opt --split-input-file --secret-to-ckks --prepare-linear-transforms --mlir-elide-elementsattrs-if-larger=4 %s | FileCheck %s

// The diagonals of a linear transform are padded to the slot count the
// ciphertext is *encoded* for, not to the ring's capacity. With logN = 14 the
// ring holds 8192 slots, but a module that asks for 1024 encodes its plaintexts
// at 1024 and a backend reads that back as LogDimensions. Padding to 8192
// instead makes this width disagree with the slot count
// prepare-linear-transforms records, which its verifier then rejects
// CHECK: @lintrans_sparse_packing
// CHECK: arith.constant dense_resource<__elided__> : tensor<2x1024xf32>
// CHECK: kernel.prepare_linear_transform
// CHECK-SAME: tensor<2x1024xf32> -> <{{.*}}slots = 1024
// CHECK-NOT: tensor<2x8192xf32>
module attributes {
  backend.lattigo,
  ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45, encryptionTechnique = extended>,
  scheme.ckks,
  scheme.actual_slot_count = 8192 : i64,
  scheme.requested_slot_count = 1024 : i64
} {
  func.func @lintrans_sparse_packing(
      %arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 45>})
      -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<2x256xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = kernel.linear_transform %input0, %cst {diagonal_indices = array<i64: 0, 1>, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : tensor<1x1024xf32>, tensor<2x256xf32> -> tensor<1x1024xf32>
      secret.yield %1 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}

// -----

// With no requested slot count the ring's capacity is the encoded width, so the
// padding target falls back to it.

// CHECK: @lintrans_full_ring
// CHECK: arith.constant dense_resource<__elided__> : tensor<2x8192xf32>
// CHECK: kernel.prepare_linear_transform
// CHECK-SAME: tensor<2x8192xf32> -> <{{.*}}slots = 8192
module attributes {
  backend.lattigo,
  ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797017456641, 35184371138561, 35184372121601], P = [1152921504607338497, 1152921504608747521], logDefaultScale = 45, encryptionTechnique = extended>,
  scheme.ckks
} {
  func.func @lintrans_full_ring(
      %arg0: !secret.secret<tensor<1x8192xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 45>})
      -> (!secret.secret<tensor<1x8192xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    %cst = arith.constant dense<1.000000e+00> : tensor<2x256xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x8192xf32>>) {
    ^body(%input0: tensor<1x8192xf32>):
      %1 = kernel.linear_transform %input0, %cst {diagonal_indices = array<i64: 0, 1>, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : tensor<1x8192xf32>, tensor<2x256xf32> -> tensor<1x8192xf32>
      secret.yield %1 : tensor<1x8192xf32>
    } -> (!secret.secret<tensor<1x8192xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %0 : !secret.secret<tensor<1x8192xf32>>
  }
}
