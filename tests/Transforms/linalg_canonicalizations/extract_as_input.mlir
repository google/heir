// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

// A flip written as tensor.extract at 3 - linalg.index reads its tensor as an
// input with that indexing map.

// CHECK: affine_map<(d0) -> (-d0 + 3)>
// CHECK: func.func @flip(%[[x:.*]]: tensor<4xf32>)
// CHECK: linalg.generic {{.*}}ins(%[[x]] : tensor<4xf32>)
// CHECK-NOT: tensor.extract
// CHECK: return
func.func @flip(%x: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%empty : tensor<4xf32>) {
  ^bb0(%out: f32):
    %i = linalg.index 0 : index
    %c3 = arith.constant 3 : index
    %j = arith.subi %c3, %i : index
    %v = tensor.extract %x[%j] : tensor<4xf32>
    linalg.yield %v : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// A lookup at an index computed from the data has no affine form and stays.

// CHECK: func.func @lookup
// CHECK: linalg.generic
// CHECK: tensor.extract
func.func @lookup(%ids: tensor<4xi32>, %table: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%ids : tensor<4xi32>) outs(%empty : tensor<4xf32>) {
  ^bb0(%id: i32, %out: f32):
    %j = arith.index_cast %id : i32 to index
    %v = tensor.extract %table[%j] : tensor<4xf32>
    linalg.yield %v : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// The verifier would require an input read at a bare loop index to have the
// loop's extent, so a longer tensor stays an extract.

// CHECK: func.func @prefix
// CHECK: tensor.extract
// CHECK: return
func.func @prefix(%x: tensor<8xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%empty : tensor<4xf32>) {
  ^bb0(%out: f32):
    %i = linalg.index 0 : index
    %v = tensor.extract %x[%i] : tensor<8xf32>
    linalg.yield %v : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// An index computed in i8 wraps at 128, which no affine map expresses, so the
// extract stays.

// CHECK: func.func @narrow_index
// CHECK: linalg.generic
// CHECK: tensor.extract
func.func @narrow_index(%x: tensor<512xf32>) -> tensor<256xf32> {
  %empty = tensor.empty() : tensor<256xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%empty : tensor<256xf32>) {
  ^bb0(%out: f32):
    %i = linalg.index 0 : index
    %n = arith.index_cast %i : index to i8
    %w = arith.index_cast %n : i8 to index
    %c256 = arith.constant 256 : index
    %j = arith.addi %w, %c256 : index
    %v = tensor.extract %x[%j] : tensor<512xf32>
    linalg.yield %v : f32
  } -> tensor<256xf32>
  return %0 : tensor<256xf32>
}

// -----

// The inner generic reads at the outer generic's loop index, which is not one
// of its own loop indices, so its extract stays.

// CHECK: func.func @outer_index(%[[x:.*]]: tensor<4xf32>)
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: tensor.extract %[[x]]
func.func @outer_index(%x: tensor<4xf32>) -> tensor<4x4xf32> {
  %empty = tensor.empty() : tensor<4x4xf32>
  %row = tensor.empty() : tensor<4xf32>
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%empty : tensor<4x4xf32>) {
  ^bb0(%out: f32):
    %i = linalg.index 0 : index
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%row : tensor<4xf32>) {
    ^bb1(%o: f32):
      %v = tensor.extract %x[%i] : tensor<4xf32>
      linalg.yield %v : f32
    } -> tensor<4xf32>
    %c0 = arith.constant 0 : index
    %e = tensor.extract %r[%c0] : tensor<4xf32>
    linalg.yield %e : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
