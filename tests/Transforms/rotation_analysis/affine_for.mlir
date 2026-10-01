// RUN: heir-opt --rotation-analysis --split-input-file %s | FileCheck %s

// The loop's yielded value is produced by an affine.for nested inside the
// scf.for. Unless affine.for is modelled, its result is resolved as an opaque
// variable and every rotation feeding it is severed from the DAG root, so
// handleScfFor reports success having found no shifts at all. Both the bare
// induction-variable rotation and the inductionVar + constant one must be
// recovered: a rolled convolution emits the pair, and the offset one is what
// aborted lattigo with "GaloisKey[39685] is nil" (39685 = 5^65).

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
