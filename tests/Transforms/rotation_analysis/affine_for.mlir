// RUN: heir-opt --rotation-analysis --split-input-file %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 1, 2, 3, 4, 5, 65, 66, 67, 68, 69>
module attributes {scheme.actual_slot_count = 128} {
  func.func @rotations_yielded_through_affine_for(
      %arg0: tensor<128xi32>) -> tensor<128xi32> {
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c64 = arith.constant 64 : index
    %0 = scf.for %i = %c1 to %c6 step %c1 iter_args(%iter = %arg0)
        -> (tensor<128xi32>) {
      %r0 = tensor_ext.rotate %arg0, %i : tensor<128xi32>, index
      %off = arith.addi %i, %c64 : index
      %r1 = tensor_ext.rotate %arg0, %off : tensor<128xi32>, index
      %sum = arith.addi %r0, %r1 : tensor<128xi32>
      %1 = affine.for %j = 0 to 2 iter_args(%acc = %sum) -> (tensor<128xi32>) {
        %2 = arith.addi %acc, %sum : tensor<128xi32>
        affine.yield %2 : tensor<128xi32>
      }
      scf.yield %1 : tensor<128xi32>
    }
    return %0 : tensor<128xi32>
  }
}

// -----

// i in [0, 5]:
//   when i - 3 >= 0 (i in [3, 5]): rotate by i       -> 3, 4, 5
//   else            (i in [0, 2]): rotate by i + 10  -> 10, 11, 12
// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 3, 4, 5, 10, 11, 12>
module attributes {scheme.actual_slot_count = 16} {
  func.func @test_affine_if(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %c10 = arith.constant 10 : index
    %0 = affine.for %i = 0 to 6 iter_args(%iter = %arg0) -> (tensor<16xi32>) {
      %1 = affine.if affine_set<(d0) : (d0 - 3 >= 0)>(%i) -> tensor<16xi32> {
        %r_then = tensor_ext.rotate %iter, %i : tensor<16xi32>, index
        affine.yield %r_then : tensor<16xi32>
      } else {
        %shift_else = arith.addi %i, %c10 : index
        %r_else = tensor_ext.rotate %iter, %shift_else : tensor<16xi32>, index
        affine.yield %r_else : tensor<16xi32>
      }
      affine.yield %1 : tensor<16xi32>
    }
    return %0 : tensor<16xi32>
  }
}

// -----

// i in [0, 7]:
//   outer if: i - 2 >= 0 (i in [2, 7])
//     x = i + 1
//     inner then: 6 - x >= 0 (i in [2, 5]) -> rotate by i      -> 2, 3, 4, 5
//     inner else: 6 - x < 0  (i in [6, 7]) -> rotate by i + 8  -> 14, 15
//   outer else: i < 2        (i in [0, 1]) -> no rotation
// CHECK: module attributes
// CHECK-SAME: rotation_analysis.indices = array<i64: 2, 3, 4, 5, 14, 15>
module attributes {scheme.actual_slot_count = 16} {
  func.func @test_nested_affine_if(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %c8 = arith.constant 8 : index
    %0 = affine.for %i = 0 to 8 iter_args(%iter = %arg0) -> (tensor<16xi32>) {
      %1 = affine.if affine_set<(d0) : (d0 - 2 >= 0)>(%i) -> tensor<16xi32> {
        %x = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
        %2 = affine.if affine_set<(d0) : (6 - d0 >= 0)>(%x) -> tensor<16xi32> {
          %r_inner_then = tensor_ext.rotate %iter, %i : tensor<16xi32>, index
          affine.yield %r_inner_then : tensor<16xi32>
        } else {
          %shift = arith.addi %i, %c8 : index
          %r_inner_else = tensor_ext.rotate %iter, %shift : tensor<16xi32>, index
          affine.yield %r_inner_else : tensor<16xi32>
        }
        affine.yield %2 : tensor<16xi32>
      } else {
        affine.yield %iter : tensor<16xi32>
      }
      affine.yield %1 : tensor<16xi32>
    }
    return %0 : tensor<16xi32>
  }
}
