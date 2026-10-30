// RUN: heir-opt --simd-vectorize-loops --canonicalize --split-input-file %s | FileCheck %s

// Each output slot reads a neighbor at a constant offset.
// CHECK: func.func @map_constant_offset
// CHECK-SAME: (%[[x:.*]]: tensor<16xi16>, %[[init:.*]]: tensor<16xi16>)
// CHECK-NOT: affine.for
// CHECK: %[[sq:.*]] = arith.muli %[[x]], %[[x]]
// CHECK: %[[rot:.*]] = tensor_ext.rotate %[[sq]]
// CHECK: return %[[rot]]
func.func @map_constant_offset(%x: tensor<16xi16>, %init: tensor<16xi16>) -> tensor<16xi16> {
  %0 = affine.for %i = 0 to 16 iter_args(%t = %init) -> tensor<16xi16> {
    %j = affine.apply affine_map<(d0) -> ((d0 + 1) mod 16)>(%i)
    %a = tensor.extract %x[%j] : tensor<16xi16>
    %b = arith.muli %a, %a : i16
    %u = tensor.insert %b into %t[%i] : tensor<16xi16>
    affine.yield %u : tensor<16xi16>
  }
  return %0 : tensor<16xi16>
}

// -----

// Only the first half of the slots is written, so the rest keeps the initial
// values.
// CHECK: func.func @map_partial
// CHECK-SAME: (%[[x:.*]]: tensor<16xi16>, %[[init:.*]]: tensor<16xi16>)
// CHECK-NOT: affine.for
// CHECK-DAG: %[[mask:.*]] = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]>
// CHECK-DAG: %[[inv:.*]] = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]>
// CHECK-DAG: %[[sum:.*]] = arith.addi %[[x]], %[[x]]
// CHECK-DAG: %[[kept:.*]] = arith.muli %[[init]], %[[inv]]
// CHECK-DAG: %[[written:.*]] = arith.muli %[[sum]], %[[mask]]
// CHECK: %[[res:.*]] = arith.addi %[[kept]], %[[written]]
// CHECK: return %[[res]]
func.func @map_partial(%x: tensor<16xi16>, %init: tensor<16xi16>) -> tensor<16xi16> {
  %0 = affine.for %i = 0 to 8 iter_args(%t = %init) -> tensor<16xi16> {
    %a = tensor.extract %x[%i] : tensor<16xi16>
    %b = arith.addi %a, %a : i16
    %u = tensor.insert %b into %t[%i] : tensor<16xi16>
    affine.yield %u : tensor<16xi16>
  }
  return %0 : tensor<16xi16>
}

// -----

// A 2D image is vectorized as its row-major flattening.
// CHECK: func.func @map_2d
// CHECK-SAME: (%[[x:.*]]: tensor<4x4xf32>, %[[init:.*]]: tensor<4x4xf32>)
// CHECK-NOT: affine.for
// CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[flat:.*]] = tensor.collapse_shape %[[x]] {{\[\[}}0, 1{{\]\]}} : tensor<4x4xf32> into tensor<16xf32>
// CHECK: %[[rot:.*]] = tensor_ext.rotate %[[flat]], %[[c4]]
// CHECK: %[[diff:.*]] = arith.subf %[[rot]], %[[flat]]
// CHECK: %[[res:.*]] = tensor.expand_shape %[[diff]]
// CHECK-SAME: tensor<16xf32> into tensor<4x4xf32>
// CHECK: return %[[res]]
func.func @map_2d(%x: tensor<4x4xf32>, %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = affine.for %r = 0 to 4 iter_args(%t0 = %init) -> tensor<4x4xf32> {
    %1 = affine.for %c = 0 to 4 iter_args(%t1 = %t0) -> tensor<4x4xf32> {
      %below = affine.apply affine_map<(d0) -> ((d0 + 1) mod 4)>(%r)
      %a = tensor.extract %x[%below, %c] : tensor<4x4xf32>
      %b = tensor.extract %x[%r, %c] : tensor<4x4xf32>
      %d = arith.subf %a, %b : f32
      %u = tensor.insert %d into %t1[%r, %c] : tensor<4x4xf32>
      affine.yield %u : tensor<4x4xf32>
    }
    affine.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// -----

// scf.for loops are supported as well.
// CHECK: func.func @map_scf
// CHECK-NOT: scf.for
// CHECK: tensor_ext.rotate
// CHECK: arith.addi
func.func @map_scf(%x: tensor<8xi32>, %init: tensor<8xi32>) -> tensor<8xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = scf.for %i = %c0 to %c8 step %c1 iter_args(%t = %init) -> tensor<8xi32> {
    %j = arith.addi %i, %c1 : index
    %k = arith.remui %j, %c8 : index
    %a = tensor.extract %x[%i] : tensor<8xi32>
    %b = tensor.extract %x[%k] : tensor<8xi32>
    %s = arith.addi %a, %b : i32
    %u = tensor.insert %s into %t[%i] : tensor<8xi32>
    scf.yield %u : tensor<8xi32>
  }
  return %0 : tensor<8xi32>
}

// -----

// A 1D box blur: the inner reduction loop sums rotations of the input and
// becomes a rotate-and-reduce.
// CHECK: func.func @stencil
// CHECK-SAME: (%[[x:.*]]: tensor<16xi16>, %[[init:.*]]: tensor<16xi16>)
// CHECK-NOT: affine.for
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c15:.*]] = arith.constant 15 : index
// CHECK: %[[r1:.*]] = tensor_ext.rotate %[[x]], %[[c1]]
// CHECK: %[[r2:.*]] = tensor_ext.rotate %[[x]], %[[c2]]
// CHECK: %[[s1:.*]] = arith.addi %[[r1]], %[[r2]]
// CHECK: %[[s2:.*]] = arith.addi %[[x]], %[[s1]]
// CHECK: %[[res:.*]] = tensor_ext.rotate %[[s2]], %[[c15]]
// CHECK: return %[[res]]
func.func @stencil(%x: tensor<16xi16>, %init: tensor<16xi16>) -> tensor<16xi16> {
  %c0_i16 = arith.constant 0 : i16
  %c16 = arith.constant 16 : index
  %0 = affine.for %i = 0 to 16 iter_args(%t = %init) -> tensor<16xi16> {
    %1 = affine.for %k = -1 to 2 iter_args(%acc = %c0_i16) -> i16 {
      %j = arith.addi %i, %k : index
      %jm = arith.remui %j, %c16 : index
      %a = tensor.extract %x[%jm] : tensor<16xi16>
      %s = arith.addi %acc, %a : i16
      affine.yield %s : i16
    }
    %u = tensor.insert %1 into %t[%i] : tensor<16xi16>
    affine.yield %u : tensor<16xi16>
  }
  return %0 : tensor<16xi16>
}

// -----

// A weighted stencil keeps the inner loop, now over whole tensors.
// CHECK: func.func @weighted_stencil
// CHECK-SAME: (%[[x:.*]]: tensor<16xi16>, %[[w:.*]]: tensor<3xi16>, %[[init:.*]]: tensor<16xi16>)
// CHECK: %[[res:.*]] = affine.for %[[k:.*]] = -1 to 2 iter_args(%[[acc:.*]] = %{{.*}}) -> (tensor<16xi16>)
// CHECK:   %[[rot:.*]] = tensor_ext.rotate %[[x]], %[[k]]
// CHECK:   %[[wk:.*]] = tensor.extract %[[w]]
// CHECK:   %[[splat:.*]] = tensor.splat %[[wk]]
// CHECK:   %[[prod:.*]] = arith.muli %[[rot]], %[[splat]]
// CHECK:   %[[sum:.*]] = arith.addi %[[acc]], %[[prod]]
// CHECK:   affine.yield %[[sum]]
// CHECK: return %[[res]]
func.func @weighted_stencil(%x: tensor<16xi16>, %w: tensor<3xi16>, %init: tensor<16xi16>) -> tensor<16xi16> {
  %c0_i16 = arith.constant 0 : i16
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %0 = affine.for %i = 0 to 16 iter_args(%t = %init) -> tensor<16xi16> {
    %1 = affine.for %k = -1 to 2 iter_args(%acc = %c0_i16) -> i16 {
      %j = arith.addi %i, %k : index
      %jm = arith.remui %j, %c16 : index
      %a = tensor.extract %x[%jm] : tensor<16xi16>
      %kw = arith.addi %k, %c1 : index
      %wk = tensor.extract %w[%kw] : tensor<3xi16>
      %p = arith.muli %a, %wk : i16
      %s = arith.addi %acc, %p : i16
      affine.yield %s : i16
    }
    %u = tensor.insert %1 into %t[%i] : tensor<16xi16>
    affine.yield %u : tensor<16xi16>
  }
  return %0 : tensor<16xi16>
}

// -----

// Taps with a constant zero weight at the ends of the window are skipped, and
// the remaining two taps are summed by a rotate_and_reduce with a constant
// table of their (splatted) weights as plaintexts.
// CHECK: func.func @weighted_stencil_zero_taps
// CHECK: arith.constant dense<{{\[\[}}3, 3, {{.*}}], [5, 5, {{.*}}]]> : tensor<2x16xi16>
// CHECK-NOT: affine.for
// CHECK: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
func.func @weighted_stencil_zero_taps(%x: tensor<16xi16>, %init: tensor<16xi16>) -> tensor<16xi16> {
  %w = arith.constant dense<[0, 3, 5]> : tensor<3xi16>
  %c0_i16 = arith.constant 0 : i16
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %0 = affine.for %i = 0 to 16 iter_args(%t = %init) -> tensor<16xi16> {
    %1 = affine.for %k = -1 to 2 iter_args(%acc = %c0_i16) -> i16 {
      %j = arith.addi %i, %k : index
      %jm = arith.remui %j, %c16 : index
      %a = tensor.extract %x[%jm] : tensor<16xi16>
      %kw = arith.addi %k, %c1 : index
      %wk = tensor.extract %w[%kw] : tensor<3xi16>
      %p = arith.muli %a, %wk : i16
      %s = arith.addi %acc, %p : i16
      affine.yield %s : i16
    }
    %u = tensor.insert %1 into %t[%i] : tensor<16xi16>
    affine.yield %u : tensor<16xi16>
  }
  return %0 : tensor<16xi16>
}

// -----

// A reduction over part of a tensor only reduces the slots of its iterations,
// leaving the result in the slot of the first one.
// CHECK: func.func @partial_sum
// CHECK-SAME: (%[[x:.*]]: tensor<8xi16>)
// CHECK-NOT: affine.for
// CHECK-NOT: arith.constant dense
// CHECK-COUNT-3: tensor_ext.rotate
// CHECK: %[[res:.*]] = tensor.extract %{{.*}}[%c2]
// CHECK: return %[[res]]
func.func @partial_sum(%x: tensor<8xi16>) -> i16 {
  %c0_i16 = arith.constant 0 : i16
  %0 = affine.for %i = 2 to 8 iter_args(%acc = %c0_i16) -> i16 {
    %a = tensor.extract %x[%i] : tensor<8xi16>
    %s = arith.addi %acc, %a : i16
    affine.yield %s : i16
  }
  return %0 : i16
}

// -----

// A product with a nontrivial initial value combines it with the reduced
// result.
// CHECK: func.func @product
// CHECK-SAME: (%[[x:.*]]: tensor<4xf32>, %[[init:.*]]: f32)
// CHECK-NOT: affine.for
// CHECK-COUNT-2: tensor_ext.rotate
// CHECK: %[[prod:.*]] = tensor.extract
// CHECK: %[[res:.*]] = arith.mulf %[[init]], %[[prod]]
// CHECK: return %[[res]]
func.func @product(%x: tensor<4xf32>, %init: f32) -> f32 {
  %0 = affine.for %i = 0 to 4 iter_args(%acc = %init) -> f32 {
    %a = tensor.extract %x[%i] : tensor<4xf32>
    %s = arith.mulf %acc, %a : f32
    affine.yield %s : f32
  }
  return %0 : f32
}

// -----

// Row sums of a secret matrix: the offset between an extracted element and the
// written slot depends on the written slot, so both loops are vectorized in
// the 16 slots of the matrix instead. The sum over each row is a
// rotate-and-reduce, which leaves it in the slot of the row's first element,
// and the four sums are moved to the output slots by one rotation and mask
// each.
// CHECK: func.func @row_sums
// CHECK-SAME: %[[m:[^:]*]]: tensor<4x4xi16> {secret.secret}
// CHECK-NOT: affine.for
// CHECK: %[[flat:.*]] = tensor.collapse_shape %[[m]]
// CHECK: %[[r1:.*]] = tensor_ext.rotate %[[flat]], %c1
// CHECK: %[[s1:.*]] = arith.addi %[[flat]], %[[r1]]
// CHECK: %[[r2:.*]] = tensor_ext.rotate %[[s1]], %c2
// CHECK: %[[sums:.*]] = arith.addi %[[s1]], %[[r2]]
// CHECK: tensor_ext.rotate %[[sums]], %c3
// CHECK: tensor_ext.rotate %[[sums]], %c6
// CHECK: tensor_ext.rotate %[[sums]], %c9
// CHECK-NOT: tensor_ext.rotate
// CHECK: %[[res:.*]] = tensor.extract_slice %{{.*}}[0] [4] [1] : tensor<16xi16> to tensor<4xi16>
// CHECK: return %[[res]]
func.func @row_sums(%m: tensor<4x4xi16> {secret.secret}, %init: tensor<4xi16>) -> tensor<4xi16> {
  %c0_i16 = arith.constant 0 : i16
  %0 = affine.for %i = 0 to 4 iter_args(%t = %init) -> tensor<4xi16> {
    %1 = affine.for %j = 0 to 4 iter_args(%acc = %c0_i16) -> i16 {
      %a = tensor.extract %m[%i, %j] : tensor<4x4xi16>
      %s = arith.addi %acc, %a : i16
      affine.yield %s : i16
    }
    %u = tensor.insert %1 into %t[%i] : tensor<4xi16>
    affine.yield %u : tensor<4xi16>
  }
  return %0 : tensor<4xi16>
}

// -----

// A plaintext tensor read at a stride is gathered into a plaintext vector
// instead of being rotated.
// CHECK: func.func @plaintext_gather
// CHECK-SAME: %[[x:[^:]*]]: tensor<8xi16> {secret.secret}, %[[w:[^:]*]]: tensor<16xi16>
// CHECK: %[[gathered:.*]] = affine.for %{{.*}} = 0 to 8 iter_args
// CHECK:   tensor.extract %[[w]]
// CHECK:   tensor.insert
// CHECK: arith.muli %[[x]], %[[gathered]] : tensor<8xi16>
// CHECK-NOT: tensor_ext.rotate
func.func @plaintext_gather(%x: tensor<8xi16> {secret.secret}, %w: tensor<16xi16>, %init: tensor<8xi16>) -> tensor<8xi16> {
  %0 = affine.for %i = 0 to 8 iter_args(%t = %init) -> tensor<8xi16> {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
    %a = tensor.extract %x[%i] : tensor<8xi16>
    %b = tensor.extract %w[%j] : tensor<16xi16>
    %c = arith.muli %a, %b : i16
    %u = tensor.insert %c into %t[%i] : tensor<8xi16>
    affine.yield %u : tensor<8xi16>
  }
  return %0 : tensor<8xi16>
}

// -----

// Gathering from a constant tensor is done at compile time.
// CHECK: func.func @constant_gather
// CHECK-SAME: %[[x:[^:]*]]: tensor<8xi16> {secret.secret}
// CHECK-NOT: affine.for
// CHECK: %[[gathered:.*]] = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14]> : tensor<8xi16>
// CHECK: arith.muli %[[x]], %[[gathered]] : tensor<8xi16>
// CHECK-NOT: tensor_ext.rotate
func.func @constant_gather(%x: tensor<8xi16> {secret.secret}, %init: tensor<8xi16>) -> tensor<8xi16> {
  %w = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi16>
  %0 = affine.for %i = 0 to 8 iter_args(%t = %init) -> tensor<8xi16> {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
    %a = tensor.extract %x[%i] : tensor<8xi16>
    %b = tensor.extract %w[%j] : tensor<16xi16>
    %c = arith.muli %a, %b : i16
    %u = tensor.insert %c into %t[%i] : tensor<8xi16>
    affine.yield %u : tensor<8xi16>
  }
  return %0 : tensor<8xi16>
}

// -----

// The same access pattern on a secret tensor needs a different rotation for
// every slot: %w[2 * i] is moved to slot i by a rotation by i and a mask
// selecting slot i.
// CHECK: func.func @secret_strided
// CHECK-SAME: %[[x:[^:]*]]: tensor<8xi16> {secret.secret}, %[[w:[^:]*]]: tensor<16xi16> {secret.secret}
// CHECK-NOT: affine.for
// CHECK-COUNT-7: tensor_ext.rotate %[[w]]
// CHECK-NOT: tensor_ext.rotate
// CHECK: %[[strided:.*]] = tensor.extract_slice %{{.*}}[0] [8] [1] : tensor<16xi16> to tensor<8xi16>
// CHECK: arith.muli %[[x]], %[[strided]]
func.func @secret_strided(%x: tensor<8xi16> {secret.secret}, %w: tensor<16xi16> {secret.secret}, %init: tensor<8xi16>) -> tensor<8xi16> {
  %0 = affine.for %i = 0 to 8 iter_args(%t = %init) -> tensor<8xi16> {
    %j = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
    %a = tensor.extract %x[%i] : tensor<8xi16>
    %b = tensor.extract %w[%j] : tensor<16xi16>
    %c = arith.muli %a, %b : i16
    %u = tensor.insert %c into %t[%i] : tensor<8xi16>
    affine.yield %u : tensor<8xi16>
  }
  return %0 : tensor<8xi16>
}

// -----

// Each iteration reads a value written by the previous one, which is not a
// map over the input.
// CHECK: func.func @prefix_sum
// CHECK: affine.for
// CHECK-NOT: tensor_ext.rotate
func.func @prefix_sum(%x: tensor<8xi16>) -> tensor<8xi16> {
  %0 = affine.for %i = 1 to 8 iter_args(%t = %x) -> tensor<8xi16> {
    %prev = affine.apply affine_map<(d0) -> (d0 - 1)>(%i)
    %a = tensor.extract %t[%prev] : tensor<8xi16>
    %b = tensor.extract %x[%i] : tensor<8xi16>
    %s = arith.addi %a, %b : i16
    %u = tensor.insert %s into %t[%i] : tensor<8xi16>
    affine.yield %u : tensor<8xi16>
  }
  return %0 : tensor<8xi16>
}

// -----

// Two iterations write the same slot.
// CHECK: func.func @not_injective
// CHECK: affine.for
// CHECK-NOT: tensor_ext.rotate
func.func @not_injective(%x: tensor<8xi16>, %init: tensor<8xi16>) -> tensor<8xi16> {
  %0 = affine.for %i = 0 to 8 iter_args(%t = %init) -> tensor<8xi16> {
    %j = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%i)
    %a = tensor.extract %x[%i] : tensor<8xi16>
    %u = tensor.insert %a into %t[%j] : tensor<8xi16>
    affine.yield %u : tensor<8xi16>
  }
  return %0 : tensor<8xi16>
}
