// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=32 | FileCheck %s

// Test layout array with two steps, should generate two loops.
#layout1 = #tensor_ext.layout<"{ [i0] -> [o0, o1] : o0 = i0 and o1 = i0 and 0 <= i0 <= 3 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and slot = i1 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }">
#composed = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = i0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 31 }">

module {
  // CHECK: func.func private @_assign_layout_{{[0-9]+}}
  // CHECK-SAME: %[[ARG0:.*]]: tensor<4xi16>) -> tensor<4x32xi16>
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[c4:.*]] = arith.constant 4 : i32
  // CHECK-DAG: %[[ALLOC2:.*]] = arith.constant dense<0> : tensor<4x32xi16>
  // CHECK-DAG: %[[ALLOC1:.*]] = arith.constant dense<0> : tensor<4x4xi16>

  // CHECK: %[[LOOP1:.*]] = scf.for %[[I0:.*]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ITER1:.*]] = %[[ALLOC1]]) -> (tensor<4x4xi16>)
  // CHECK:   %[[IDX0:.*]] = arith.index_cast %[[I0]] : i32 to index
  // CHECK:   %[[EXT1:.*]] = tensor.extract %[[ARG0]][%[[IDX0]]]
  // CHECK:   %[[INS1:.*]] = tensor.insert %[[EXT1]] into %[[ITER1]][{{.*}}, {{.*}}]
  // CHECK:   scf.yield %[[INS1]]

  // CHECK: %[[LOOP2:.*]] = scf.for %[[I0_2:.*]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ITER2:.*]] = %[[ALLOC2]]) -> (tensor<4x32xi16>)
  // CHECK:   %[[LOOP3:.*]] = scf.for %[[I1_2:.*]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ITER3:.*]] = %[[ITER2]]) -> (tensor<4x32xi16>)
  // CHECK:     %[[EXT2:.*]] = tensor.extract %[[LOOP1]][{{.*}}, {{.*}}]
  // CHECK:     %[[INS2:.*]] = tensor.insert %[[EXT2]] into %[[ITER3]][{{.*}}, {{.*}}]
  // CHECK:     scf.yield %[[INS2]]
  // CHECK:   scf.yield %[[LOOP3]]

  // CHECK: func.func @assign_layout_array_i16
  func.func @assign_layout_array_i16() -> (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #composed}) {
    %cst = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi16>
    // CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi16>
    // CHECK: %[[GEN:.*]] = secret.generic()
    // CHECK:   %[[CALL:.*]] = func.call @_assign_layout_{{[0-9]+}}(%[[CST]]) : (tensor<4xi16>) -> tensor<4x32xi16>
    // CHECK:   secret.yield %[[CALL]] : tensor<4x32xi16>

    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {
        layout = [#layout1, #layout2],
        tensor_ext.layout = #composed
      } : tensor<4xi16>
      secret.yield %1 : tensor<4xi16>
    } -> (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #composed})
    // CHECK: return %[[GEN]]
    return %0 : !secret.secret<tensor<4xi16>>
  }
}

// -----

// 3 steps in layout, including a larger intermediate type (3-D tensor)
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [o0, o1, o2] : o0 = i0 and o1 = i1 and o2 = i1 and 0 <= i0 <= 1 and 0 <= i1 <= 1 }">
#layout2 = #tensor_ext.layout<"{ [o0, o1, o2] -> [p0, p1, p2] : p0 = o1 and p1 = o0 and p2 = o2 and 0 <= o0 <= 1 and 0 <= o1 <= 1 and 0 <= o2 <= 1 }">
#layout3 = #tensor_ext.layout<"{ [p0, p1, p2] -> [ct, slot] : ct = p0 and slot = p1 * 2 + p2 and 0 <= p0 <= 1 and 0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 <= slot <= 31 }">
#composed = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i1 and slot = i0 * 2 + i1 and 0 <= i0 <= 1 and 0 <= i1 <= 1 and 0 <= slot <= 31 }">

module {
  // CHECK: func.func private @_assign_layout_{{[0-9]+}}
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x2xi32>) -> tensor<2x32xi32>
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : i32
  // CHECK-DAG: %[[ALLOC3:.*]] = arith.constant dense<0> : tensor<2x32xi32>
  // CHECK-DAG: %[[ALLOC1:.*]] = arith.constant dense<0> : tensor<2x2x2xi32>

  // CHECK: %[[LOOP1:.*]] = scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ITER1:.*]] = %[[ALLOC1]]) -> (tensor<2x2x2xi32>){{.*}}
  // CHECK:   scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ITER1_2:.*]] = %[[ITER1]]) -> (tensor<2x2x2xi32>){{.*}}
  // CHECK:     tensor.extract %[[ARG0]][{{.*}}, {{.*}}]
  // CHECK:     tensor.insert %{{.*}} into %[[ITER1_2]][{{.*}}, {{.*}}, {{.*}}]

  // CHECK: %[[LOOP2:.*]] = scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ITER2:.*]] = %[[ALLOC1]]) -> (tensor<2x2x2xi32>){{.*}}
  // CHECK:   scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ITER2_2:.*]] = %[[ITER2]]) -> (tensor<2x2x2xi32>){{.*}}
  // CHECK:     scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ITER2_3:.*]] = %[[ITER2_2]]) -> (tensor<2x2x2xi32>){{.*}}
  // CHECK:       tensor.extract %[[LOOP1]][{{.*}}, {{.*}}, {{.*}}]
  // CHECK:       tensor.insert %{{.*}} into %[[ITER2_3]][{{.*}}, {{.*}}, {{.*}}]

  // CHECK: %[[LOOP3:.*]] = scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ITER3:.*]] = %[[ALLOC3]]) -> (tensor<2x32xi32>){{.*}}
  // CHECK:   scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ITER3_2:.*]] = %[[ITER3]]) -> (tensor<2x32xi32>){{.*}}
  // CHECK:     scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ITER3_3:.*]] = %[[ITER3_2]]) -> (tensor<2x32xi32>){{.*}}
  // CHECK:       tensor.extract %[[LOOP2]][{{.*}}, {{.*}}, {{.*}}]
  // CHECK:       tensor.insert %{{.*}} into %[[ITER3_3]][{{.*}}, {{.*}}]

  // CHECK: func.func @assign_layout_array_3step
  func.func @assign_layout_array_3step() -> (!secret.secret<tensor<2x2xi32>> {tensor_ext.layout = #composed}) {
    %cst = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>
    // CHECK: %[[CST:.*]] = arith.constant dense<{{\[\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
    // CHECK: %[[GEN:.*]] = secret.generic()
    // CHECK:   %[[CALL:.*]] = func.call @_assign_layout_{{[0-9]+}}(%[[CST]]) : (tensor<2x2xi32>) -> tensor<2x32xi32>
    // CHECK:   secret.yield %[[CALL]] : tensor<2x32xi32>

    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {
        layout = [#layout1, #layout2, #layout3],
        tensor_ext.layout = #composed
      } : tensor<2x2xi32>
      secret.yield %1 : tensor<2x2xi32>
    } -> (!secret.secret<tensor<2x2xi32>> {tensor_ext.layout = #composed})
    // CHECK: return %[[GEN]]
    return %0 : !secret.secret<tensor<2x2xi32>>
  }
}

// -----

// zero valued input
#layout1 = #tensor_ext.layout<"{ [i0] -> [o0, o1] : o0 = i0 and o1 = i0 and 0 <= i0 <= 3 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and slot = i1 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= slot <= 31 }">
#composed = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = i0 and slot = i0 and 0 <= i0 <= 3 and 0 <= slot <= 31 }">

module {
  // CHECK: func.func @assign_layout_array_zero
  func.func @assign_layout_array_zero() -> (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #composed}) {
    %cst = arith.constant dense<0> : tensor<4xi16>
    // CHECK: %[[CST:.*]] = arith.constant dense<0> : tensor<4x32xi16>
    // CHECK: %[[GEN:.*]] = secret.generic()
    // CHECK:   secret.yield %[[CST]] : tensor<4x32xi16>

    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {
        layout = [#layout1, #layout2],
        tensor_ext.layout = #composed
      } : tensor<4xi16>
      secret.yield %1 : tensor<4xi16>
    } -> (!secret.secret<tensor<4xi16>> {tensor_ext.layout = #composed})
    // CHECK: return %[[GEN]]
    return %0 : !secret.secret<tensor<4xi16>>
  }
}
