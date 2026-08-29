// RUN: heir-opt --implement-shift-network %s | FileCheck %s

// When the permutation is itself a cyclic shift, the resulting shift network
// should also have a single shift.
#map1 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : ct1 = 0 and ct2 = 0 and ((slot1 - 1) - slot2) mod 64 = 0 and slot1 >= 0 and 63 >= slot1 and slot2 >= 0 and 63 >= slot2 }">
// CHECK: func.func @test_no_conflicts
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x64xi32>) -> tensor<1x64xi32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> to tensor<1x64xi32>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[ROT:.*]] = tensor_ext.rotate %[[SLICE]], %[[C1]] : tensor<1x64xi32>, index
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[ROT]] into %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> into tensor<1x64xi32>
// CHECK: return %[[INSERT]] : tensor<1x64xi32>
func.func @test_no_conflicts(%0: tensor<1x64xi32>) -> tensor<1x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map1} : tensor<1x64xi32>
  return %1 : tensor<1x64xi32>
}

// When the permutation is a single rotation, direct depth-1 rotation is
// selected, emitting a single rotation by 63 (or -1).
//
// CHECK: func.func @test_no_conflicts2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x64xi32>) -> tensor<1x64xi32>
// CHECK: %[[SLICE2:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> to tensor<1x64xi32>
// CHECK: %[[C63:.*]] = arith.constant 63 : index
// CHECK: %[[ROT2:.*]] = tensor_ext.rotate %[[SLICE2]], %[[C63]] : tensor<1x64xi32>, index
// CHECK: %[[INSERT2:.*]] = tensor.insert_slice %[[ROT2]] into %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> into tensor<1x64xi32>
// CHECK: return %[[INSERT2]] : tensor<1x64xi32>
#map2 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : ct1 = 0 and ct2 = 0 and ((slot1 + 1) - slot2) mod 64 = 0 and slot1 >= 0 and 63 >= slot1 and slot2 >= 0 and 63 >= slot2 }">
func.func @test_no_conflicts2(%0: tensor<1x64xi32>) -> tensor<1x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map2} : tensor<1x64xi32>
  return %1 : tensor<1x64xi32>
}


// CHECK: func.func @identity
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x64xi32>) -> tensor<1x64xi32>
// CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> to tensor<1x64xi32>
// CHECK: %[[INSERT:.*]] = tensor.insert_slice %[[SLICE]] into %[[ARG0]][0, 0] [1, 64] [1, 1] : tensor<1x64xi32> into tensor<1x64xi32>
// CHECK: return %[[INSERT]] : tensor<1x64xi32>
#map3 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : ct1 = 0 and ct2 = 0 and slot1 = slot2 and slot1 >= 0 and 63 >= slot1 and slot2 >= 0 and 63 >= slot2 }">
func.func @identity(%0: tensor<1x64xi32>) -> tensor<1x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map3} : tensor<1x64xi32>
  return %1 : tensor<1x64xi32>
}

// CHECK: func.func @multi_ciphertext_swap_cts
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x64xi32>) -> tensor<4x64xi32>
// CHECK-DAG: %[[SLICE0:.*]] = tensor.extract_slice %[[ARG0]][0, 0] [1, 64] [1, 1]
// CHECK-DAG: %[[SLICE1:.*]] = tensor.extract_slice %[[ARG0]][1, 0] [1, 64] [1, 1]
// CHECK-DAG: %[[SLICE2:.*]] = tensor.extract_slice %[[ARG0]][2, 0] [1, 64] [1, 1]
// CHECK-DAG: %[[SLICE3:.*]] = tensor.extract_slice %[[ARG0]][3, 0] [1, 64] [1, 1]
// CHECK: %[[INSERT0:.*]] = tensor.insert_slice %[[SLICE3]] into %[[ARG0]][0, 0]
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
// CHECK-COUNT-4: tensor_ext.rotate
#map5 = #tensor_ext.layout<"{ [ct1, slot1] -> [ct2, slot2] : (ct1 - ct2) mod 4 = 3 and (slot1 - slot2) mod 64 = 5 and 0 <= ct1 <= 3 and 0 <= ct2 <= 3 and 0 <= slot1 <= 63 and 0 <= slot2 <= 63 }">
func.func @multi_ciphertext_complex(%0: tensor<4x64xi32>) -> tensor<4x64xi32> {
  %1 = tensor_ext.remap %0 {permutation = #map5} : tensor<4x64xi32>
  return %1 : tensor<4x64xi32>
}

// CHECK: func.func @periodic_replication
// CHECK-NOT: tensor_ext.remap
// CHECK: tensor.extract_slice
// CHECK: tensor_ext.rotate
// CHECK: tensor.insert_slice
#layout_bicyclic = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (4i0 + 5i1 + slot) mod 30 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 1 and 0 <= slot <= 1023 }">
#replication = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 6 = 0 and 0 <= i1 <= 5 and 0 <= slot <= 1023 }">
func.func @periodic_replication(%arg0: tensor<1x1024xi16> {tensor_ext.layout = #layout_bicyclic}) -> (tensor<1x1024xi16> {tensor_ext.layout = #layout_bicyclic}) {
  %0 = tensor_ext.remap %arg0 {permutation = #replication} : tensor<1x1024xi16>
  return %0 : tensor<1x1024xi16>
}
