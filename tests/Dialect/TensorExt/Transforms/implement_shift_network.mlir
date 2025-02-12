// RUN: heir-opt --implement-shift-network=ciphertext-size=64 %s | FileCheck %s

// When the permutation is itself a cyclic shift, the resulting shift network
// should also have a single shift.
#map1 = affine_map<(d0) -> ((d0 - 1) mod 64)>
// CHECK-LABEL: @test_no_conflicts
// CHECK-SAME: [[arg0:%[^:]*]]: tensor<64xi32>
// CHECK-NEXT: [[mask:%[^:]*]] = arith.constant dense<1>
// CHECK-NEXT: [[mul_mask:%[^:]*]] = arith.muli [[arg0]], [[mask]]
// CHECK-NEXT: [[rot_amt:%[^:]*]] = arith.constant 1
// CHECK-NEXT: [[rot_result:%[^:]*]] = tensor_ext.rotate [[mul_mask]], [[rot_amt]]
// CHECK-NEXT: return [[rot_result]]
func.func @test_no_conflicts(%0: tensor<64xi32>) -> tensor<64xi32> {
  %1 = tensor_ext.permute %0 {permutation = #map1} : tensor<64xi32>
  return %1 : tensor<64xi32>
}

// This test has a larger set of rotations because the Vos-Vos-Erkin method
// forces each rotation to be decomposed into power-of-two rotations, even if
// it could be done in a single rotation. In this case it's a (left-)rotation
// by 63 which is equivalent to a single (right)-rotation by -1, which requires
// all power-of-two components to be used.
//
// TODO(#744): perhaps this test should only produce one rotation.
// CHECK-LABEL: @test_no_conflicts_2
// CHECK-SAME: [[arg0:%[^:]*]]: tensor<64xi32>

// Rot by 1
// CHECK-NEXT: [[mask1:%[^:]*]] = arith.constant dense<1>
// CHECK-NEXT: [[mul_mask1:%[^:]*]] = arith.muli [[arg0]], [[mask1]]
// CHECK-NEXT: [[rot_amt1:%[^:]*]] = arith.constant 1
// CHECK-NEXT: [[rot_result1:%[^:]*]] = tensor_ext.rotate [[mul_mask1]], [[rot_amt1]]

// Rot by 2
// CHECK-NEXT: [[mask2:%[^:]*]] = arith.constant dense<1>
// CHECK-NEXT: [[mul_mask2:%[^:]*]] = arith.muli [[arg0]], [[mask2]]
// CHECK-NEXT: [[rot_amt2:%[^:]*]] = arith.constant 2
// CHECK-NEXT: [[rot_result2:%[^:]*]] = tensor_ext.rotate [[mul_mask2]], [[rot_amt2]]
// CHECK-NEXT: [[combined_result2:%[^:]*]] = arith.addi [[rot_result1]], [[rot_result2]]

// Rot by 4
// CHECK-NEXT: [[mask4:%[^:]*]] = arith.constant dense<1>
// CHECK-NEXT: [[mul_mask4:%[^:]*]] = arith.muli [[arg0]], [[mask4]]
// CHECK-NEXT: [[rot_amt4:%[^:]*]] = arith.constant 4
// CHECK-NEXT: [[rot_result4:%[^:]*]] = tensor_ext.rotate [[mul_mask4]], [[rot_amt4]]
// CHECK-NEXT: [[combined_result4:%[^:]*]] = arith.addi [[combined_result2]], [[rot_result4]]

// Rot by 8
// CHECK-NEXT: [[mask8:%[^:]*]] = arith.constant dense<1>
// CHECK-NEXT: [[mul_mask8:%[^:]*]] = arith.muli [[arg0]], [[mask8]]
// CHECK-NEXT: [[rot_amt8:%[^:]*]] = arith.constant 8
// CHECK-NEXT: [[rot_result8:%[^:]*]] = tensor_ext.rotate [[mul_mask8]], [[rot_amt8]]
// CHECK-NEXT: [[combined_result8:%[^:]*]] = arith.addi [[combined_result4]], [[rot_result8]]

// Rot by 16
// CHECK-NEXT: [[mask16:%[^:]*]] = arith.constant dense<1>
// CHECK-NEXT: [[mul_mask16:%[^:]*]] = arith.muli [[arg0]], [[mask16]]
// CHECK-NEXT: [[rot_amt16:%[^:]*]] = arith.constant 16
// CHECK-NEXT: [[rot_result16:%[^:]*]] = tensor_ext.rotate [[mul_mask16]], [[rot_amt16]]
// CHECK-NEXT: [[combined_result16:%[^:]*]] = arith.addi [[combined_result8]], [[rot_result16]]

// Rot by 32
// CHECK-NEXT: [[mask32:%[^:]*]] = arith.constant dense<1>
// CHECK-NEXT: [[mul_mask32:%[^:]*]] = arith.muli [[arg0]], [[mask32]]
// CHECK-NEXT: [[rot_amt32:%[^:]*]] = arith.constant 32
// CHECK-NEXT: [[rot_result32:%[^:]*]] = tensor_ext.rotate [[mul_mask32]], [[rot_amt32]]
// CHECK-NEXT: [[combined_result32:%[^:]*]] = arith.addi [[combined_result16]], [[rot_result32]]

// CHECK-NEXT: return [[combined_result32]]
#map2 = affine_map<(d0) -> ((d0 + 1) mod 64)>
func.func @test_no_conflicts_2(%0: tensor<64xi32>) -> tensor<64xi32> {
  %1 = tensor_ext.permute %0 {permutation = #map2} : tensor<64xi32>
  return %1 : tensor<64xi32>
}


// This test is similar to Figure 3 from the Vos-Vos-Erkin paper, but with the
// indices outside of 1..16 being fixed to the identity permutation. This makes
// the graph significantly more complicated, since indices that shift are
// passing through many indices that are fixed points. I think it makes for a
// nice example that exercises many of the edge cases of the algorithm.
//
// CHECK-LABEL: @figure3
//
// Group:  2 4 6 7 10 12 14 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 37 38 39 40 41 42 45 47 48 49 50 51 53 54 55 56 57 58 61 63
// Shifts: 0 0 62 0 57 0 56 2 0 0 62 0 2 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Indices shifted by 1:  4 14
// CHECK: arith.constant dense<[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot1_1:%[^ ]*]] = arith.constant 1
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot1_1]]
// Indices shifted by 2:  2 7 10 12
// CHECK: arith.constant dense<[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot1_2:%[^ ]*]] = arith.constant 2
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot1_2]]
// Indices shifted by 4:  2 10 14
// CHECK: arith.constant dense<[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot1_4:%[^ ]*]] = arith.constant 4
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot1_4]]
// Indices shifted by 8:  2 4 6 10
// CHECK: arith.constant dense<[0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot1_8:%[^ ]*]] = arith.constant 8
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot1_8]]
// Indices shifted by 16: 2 4 6 10
// CHECK: arith.constant dense<[0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot1_16:%[^ ]*]] = arith.constant 16
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot1_16]]
// Indices shifted by 32: 2 4 6 10
// CHECK: arith.constant dense<[0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot1_32:%[^ ]*]] = arith.constant 32
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot1_32]]
//
// Group:  0 1 5 8 11 15 36 43 44 46 52 59 60 62
// Shifts: 51 57 0 0 0 62 0 0 57 0 0 5 0 0 0 14 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Indices shifted by 1:  0 1 8 11
// CHECK: arith.constant dense<[1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot2_1:%[^ ]*]] = arith.constant 1
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot2_1]]
// Indices shifted by 2:  0 5 15
// CHECK: arith.constant dense<[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot2_2:%[^ ]*]] = arith.constant 2
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot2_2]]
// Indices shifted by 4:  5 11 15
// CHECK: arith.constant dense<[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot2_4:%[^ ]*]] = arith.constant 4
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot2_4]]
// Indices shifted by 8:  1 5 8 15
// CHECK: arith.constant dense<[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot2_8:%[^ ]*]] = arith.constant 8
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot2_8]]
// Indices shifted by 16: 0 1 5 8
// CHECK: arith.constant dense<[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot2_16:%[^ ]*]] = arith.constant 16
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot2_16]]
// Indices shifted by 32: 0 1 5 8
// CHECK: arith.constant dense<[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot2_32:%[^ ]*]] = arith.constant 32
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot2_32]]
//
// Group:  3 9 13
// Shifts: 0 0 0 3 0 0 0 0 0 6 0 0 0 11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// Indices shifted by 1:  3 13
// CHECK: arith.constant dense<[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot3_1:%[^ ]*]] = arith.constant 1
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot3_1]]
// Indices shifted by 2:  3 9 13
// CHECK: arith.constant dense<[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot3_2:%[^ ]*]] = arith.constant 2
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot3_2]]
// Indices shifted by 4:  9
// CHECK: arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot3_4:%[^ ]*]] = arith.constant 4
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot3_4]]
// Indices shifted by 8:  13
// CHECK: arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK: [[rot3_8:%[^ ]*]] = arith.constant 8
// CHECK: tensor_ext.rotate
// CHECK-SAME: [[rot3_8]]
// Indices shifted by 16:
// Indices shifted by 32:
// CHECK-NOT: tensor_ext.rotate
#map3 = dense<[13, 8, 4, 0, 11, 7, 14, 5, 15, 3, 12, 6, 10, 2, 9, 1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
func.func @figure3(%0: tensor<64xi32>) -> tensor<64xi32> {
  %1 = tensor_ext.permute %0 {permutation = #map3} : tensor<64xi32>
  return %1 : tensor<64xi32>
}


// CHECK-LABEL: func.func @identity
// CHECK-NEXT: return
#identityperm = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
func.func @identity(%0: tensor<64xi32>) -> tensor<64xi32> {
  %1 = tensor_ext.permute %0 {permutation = #identityperm} : tensor<64xi32>
  return %1 : tensor<64xi32>
}
