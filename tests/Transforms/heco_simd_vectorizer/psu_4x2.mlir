// Ported from https://github.com/MarbleHE/HECO/blob/3e13744233ab0c09030a41ef98b4e061b6fa2eac/test/filecheck/psu_small/psu_1_ssa.mlir

// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse --heco-simd-vectorizer %s | FileCheck %s

// The sum over %a_data is a rotate-and-reduce (two rotations). The duplicate
// check is vectorized over the rows of %a_id and their two columns, in the 8
// slots of %a_id, while the loop over the rows of %b_id remains a loop that is
// reindexed to rotate %b_id by 2 * t in iteration t. The products over the
// columns are rotate-and-reduces (one rotation each), %b_data is spread to the
// slots of the rows (three rotations), and the final sum is a
// rotate-and-reduce over the rows (two rotations). The original HECO needs 29
// rotations.
// CHECK: @psu
// CHECK-COUNT-2: tensor_ext.rotate
// CHECK: affine.for %{{.*}} = 0 to 4 iter_args(%{{.*}}) -> (tensor<8xi16>)
// CHECK-COUNT-2: tensor_ext.rotate
// CHECK: affine.yield
// CHECK-COUNT-5: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
// CHECK-NOT: affine.for

func.func @psu(%a_id: tensor<4x2xi16>, %a_data: tensor<4xi16>, %b_id: tensor<4x2xi16>, %b_data: tensor<4xi16>) -> i16 {
  %c0_si16 = arith.constant 0 : i16
  %c1_si16 = arith.constant 1 : i16

  // First, sum all of A
  %sum_a = affine.for %i = 0 to 4 iter_args(%cur0 = %c0_si16) -> (i16) {
    %1 = tensor.extract %a_data[%i] : tensor<4xi16>
    %2 = arith.addi %cur0, %1 : i16
    affine.yield %2 : i16
  }

  // Now sum only those of B that aren't duplicates of A
  %sum = affine.for %i = 0 to 4 iter_args(%cur_sum = %sum_a) -> (i16) {
    // check if a[i] is a dupe of b[j]
    %unique = affine.for %j = 0 to 4 iter_args(%cur_unique = %c1_si16) -> i16 {
      // compute a_id[i] == b_id[j]
      %equal = affine.for %k = 0 to 2 iter_args(%cur_equal = %c1_si16) -> i16 {
        %6 = tensor.extract %a_id[%i, %k] : tensor<4x2xi16>
        %7 = tensor.extract %b_id[%j, %k] : tensor<4x2xi16>
        %8 = arith.subi %6, %7 : i16
        %9 = arith.muli %8, %8 : i16
        %10 = arith.subi %c1_si16, %9 : i16
        %11 = arith.muli %10, %cur_equal : i16
        affine.yield %11 : i16
      }
      %nequal = arith.subi %c1_si16, %equal : i16
      %13 = arith.muli %cur_unique, %nequal : i16
      affine.yield %13 : i16
    }
    %14 = tensor.extract %b_data[%i] : tensor<4xi16>
    %15 = arith.muli %14, %unique : i16
    %16 = arith.addi %cur_sum, %15 : i16
    affine.yield %16 : i16
  }
  return %sum : i16
}
