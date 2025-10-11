// RUN: heir-opt --implement-shift-network %s | FileCheck %s

// When the permutation is itself a cyclic shift, the resulting shift network
// should also have a single shift.
#map1 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : ct1 = 0 and ct2 = 0 and ((slot1 - 1) - slot2) mod 64 = 0 and slot1 >= 0 and 63 >= slot1 and slot2 >= 0 and 63 >= slot2 }">
// CHECK: func.func @test_no_conflicts
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x64xi32>) -> tensor<1x64xi32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> to tensor<64xi32>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[ROT:.*]] = tensor_ext.rotate %[[SLICE]], %[[C1]] : tensor<64xi32>, index
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x64xi32>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[ROT]] into %[[EMPTY]][0, 0] [1, 64] [1, 1] : tensor<64xi32> into tensor<1x64xi32>
// CHECK: return %[[INSERT]] : tensor<1x64xi32>
func.func @test_no_conflicts(%0: tensor<1x64xi32>) -> tensor<1x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map1} : tensor<1x64xi32>
  return %1 : tensor<1x64xi32>
}

// This test has a larger set of rotations because the Vos-Vos-Erkin method
// forces each rotation to be decomposed into power-of-two rotations, even if
// it could be done in a single rotation. In this case it's a (left-)rotation
// by 63 which is equivalent to a single (right)-rotation by -1, which requires
// all power-of-two components to be used.
//
// TODO(#2263): this test should only produce one rotation by -1
//
// CHECK: func.func @test_no_conflicts2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x64xi32>) -> tensor<1x64xi32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> to tensor<64xi32>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[ROT0:.*]] = tensor_ext.rotate %[[SLICE]], %[[C1]] : tensor<64xi32>, index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[ROT1:.*]] = tensor_ext.rotate %[[ROT0]], %[[C2]] : tensor<64xi32>, index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[ROT2:.*]] = tensor_ext.rotate %[[ROT1]], %[[C4]] : tensor<64xi32>, index
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[ROT3:.*]] = tensor_ext.rotate %[[ROT2]], %[[C8]] : tensor<64xi32>, index
// CHECK: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[ROT4:.*]] = tensor_ext.rotate %[[ROT3]], %[[C16]] : tensor<64xi32>, index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[ROT5:.*]] = tensor_ext.rotate %[[ROT4]], %[[C32]] : tensor<64xi32>, index
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x64xi32>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[ROT5]] into %[[EMPTY]][0, 0] [1, 64] [1, 1] : tensor<64xi32> into tensor<1x64xi32>
// CHECK: return %[[INSERT]] : tensor<1x64xi32>
#map2 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : ct1 = 0 and ct2 = 0 and ((slot1 + 1) - slot2) mod 64 = 0 and slot1 >= 0 and 63 >= slot1 and slot2 >= 0 and 63 >= slot2 }">
func.func @test_no_conflicts2(%0: tensor<1x64xi32>) -> tensor<1x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map2} : tensor<1x64xi32>
  return %1 : tensor<1x64xi32>
}


// CHECK: func.func @identity
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x64xi32>) -> tensor<1x64xi32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> to tensor<64xi32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1x64xi32>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[SLICE]] into %[[EMPTY]][0, 0] [1, 64] [1, 1] : tensor<64xi32> into tensor<1x64xi32>
// CHECK: return %[[INSERT]] : tensor<1x64xi32>
#map3 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : ct1 = 0 and ct2 = 0 and slot1 = slot2 and slot1 >= 0 and 63 >= slot1 and slot2 >= 0 and 63 >= slot2 }">
func.func @identity(%0: tensor<1x64xi32>) -> tensor<1x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map3} : tensor<1x64xi32>
  return %1 : tensor<1x64xi32>
}

// CHECK: func.func @multi_ciphertext_swap_cts
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x64xi32>) -> tensor<4x64xi32>
// CHECK-DAG: %[[SLICE0:.*]] = tensor.extract_slice %arg0[0, 0] [1, 64] [1, 1]
// CHECK-DAG: %[[SLICE1:.*]] = tensor.extract_slice %arg0[1, 0] [1, 64] [1, 1]
// CHECK-DAG: %[[SLICE2:.*]] = tensor.extract_slice %arg0[2, 0] [1, 64] [1, 1]
// CHECK-DAG: %[[SLICE3:.*]] = tensor.extract_slice %arg0[3, 0] [1, 64] [1, 1]
// CHECK: %[[EMPTY:.*]] = tensor.empty()
// CHECK-NEXT: %[[INSERT0:.*]] = tensor.insert_slice %[[SLICE3]] into %[[EMPTY]][0, 0]
// CHECK-NEXT: %[[INSERT1:.*]] = tensor.insert_slice %[[SLICE0]] into %[[INSERT0]][1, 0]
// CHECK-NEXT: %[[INSERT2:.*]] = tensor.insert_slice %[[SLICE1]] into %[[INSERT1]][2, 0]
// CHECK-NEXT: %[[INSERT3:.*]] = tensor.insert_slice %[[SLICE2]] into %[[INSERT2]][3, 0]
// CHECK-NEXT: return %[[INSERT3]]
#map4 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : (ct1 - ct2) mod 4 = 3 and (slot1 - slot2) mod 64 = 0 and 0 <= ct1 <= 3 and 0 <= ct2 <= 3 and 0 <= slot1 <= 63 and 0 <= slot2 <= 63 }">
func.func @multi_ciphertext_swap_cts(%0: tensor<4x64xi32>) -> tensor<4x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map4} : tensor<4x64xi32>
  return %1 : tensor<4x64xi32>
}

// Not testing the correctness of the shift network (see
// ImplementShiftNetworkTest.cpp for that), just that the IR materializes
// properly with multi-ciphertext inputs.
//
// CHECK: func.func @multi_ciphertext_complex
// CHECK-COUNT-16: tensor_ext.rotate
#map5 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : (ct1 - ct2) mod 4 = 3 and (slot1 - slot2) mod 64 = 5 and 0 <= ct1 <= 3 and 0 <= ct2 <= 3 and 0 <= slot1 <= 63 and 0 <= slot2 <= 63 }">
func.func @multi_ciphertext_complex(%0: tensor<4x64xi32>) -> tensor<4x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map5} : tensor<4x64xi32>
  return %1 : tensor<4x64xi32>
}
