// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=32 | FileCheck %s


// Test that a vector of size 16xi16 is replicated to 1x32xi16.
// CHECK: func.func private @_assign_layout_{{[0-9]+}}
// CHECK-SAME: %[[ARG0:.*]]: tensor<16xi16>) -> tensor<1x32xi16>
// CHECK-DAG: %[[c16:.*]] = arith.constant 16 : i32
// CHECK-DAG: %[[c32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[cst:.*]] = arith.constant dense<0> : tensor<1x32xi16>
// CHECK: scf.for %[[arg1:.*]] = %[[c0]] to %[[c32]] step %[[c1]]
// CHECK: tensor.insert

// CHECK: @repeat_vector
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 31 }">
module {
  func.func @repeat_vector() {
    %cst = arith.constant dense<1> : tensor<16xi16>
    // CHECK: %[[cst:.*]] = arith.constant dense<1> : tensor<16xi16>
    // CHECK: func.call @_assign_layout_{{[0-9]+}}(%[[cst]])
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<16xi16>
      secret.yield %1 : tensor<16xi16>
    } -> (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #layout})
    return
  }
}

// -----

#layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 31 }">
module {
  // CHECK: @scalar_mul
  func.func @scalar_mul(%arg0: !secret.secret<i16> {tensor_ext.layout = #layout}) -> (!secret.secret<i16> {tensor_ext.layout = #layout}) {
    // Dropping unit dims results in a 32xi16 tensor for assign layout
    // CHECK: %[[cst:.*]] = arith.constant dense<2> : tensor<1x32xi16>
    %0 = secret.generic(%arg0: !secret.secret<i16> {tensor_ext.layout = #layout}) {
    ^body(%input0: i16):
      %c2_i16 = arith.constant 2 : i16
      %1 = tensor_ext.assign_layout %c2_i16 {layout = #layout, tensor_ext.layout = #layout} : i16
      %2 = arith.muli %input0, %1 {tensor_ext.layout = #layout} : i16
      secret.yield %2 : i16
    } -> (!secret.secret<i16> {tensor_ext.layout = #layout})
    // CHECK: return {{.*}} : !secret.secret<tensor<1x32xi16>>
    return %0 : !secret.secret<i16>
  }
}

// -----

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 16 = 0 and 0 <= i0 <= 15 and 0 <= slot <= 31 }">
module {
  // CHECK: @empty
  func.func @empty() -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) {
    %0 = secret.generic() {
      // CHECK: %[[empty:.*]] = tensor.empty() : tensor<1x32xi16>
      %empty = tensor.empty() : tensor<32xi16>
      %1 = tensor_ext.assign_layout %empty {layout = #layout, tensor_ext.layout = #layout} : tensor<32xi16>
      secret.yield %1 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout})
    // CHECK: secret.yield %[[empty]] : tensor<1x32xi16>
    // CHECK: return {{.*}} : !secret.secret<tensor<1x32xi16>>
    return %0 : !secret.secret<tensor<32xi16>>
  }
}

// -----

// CHECK: func.func private @_assign_layout_{{[0-9]+}}
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x32xi16>) -> tensor<1x32xi16>
// CHECK-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[c5:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[c6:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
// CHECK-DAG: %[[cst:.*]] = arith.constant dense<0> : tensor<1x32xi16>
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[EXT0:.*]] = tensor.extract %arg0[%[[c0]], %[[c0]]]
// CHECK-DAG: %[[INS0:.*]] = tensor.insert %[[EXT0]] into %[[cst]][%[[c0]], %[[c7]]]
// CHECK-DAG: %[[EXT1:.*]] = tensor.extract %arg0[%[[c0]], %[[c1]]]
// CHECK-DAG: %[[INS1:.*]] = tensor.insert %[[EXT1]] into %[[INS0]][%[[c0]], %[[c6]]]
// CHECK-DAG: %[[EXT2:.*]] = tensor.extract %arg0[%[[c0]], %[[c2]]]
// CHECK-DAG: %[[INS2:.*]] = tensor.insert %[[EXT2]] into %[[INS1]][%[[c0]], %[[c5]]]
// CHECK-DAG: %[[EXT3:.*]] = tensor.extract %arg0[%[[c0]], %[[c3]]]
// CHECK-DAG: %[[INS3:.*]] = tensor.insert %[[EXT3]] into %[[INS2]][%[[c0]], %[[c4]]]
// CHECK-DAG: %[[EXT4:.*]] = tensor.extract %arg0[%[[c0]], %[[c4]]]
// CHECK-DAG: %[[INS4:.*]] = tensor.insert %[[EXT4]] into %[[INS3]][%[[c0]], %[[c3]]]
// CHECK-DAG: %[[EXT5:.*]] = tensor.extract %arg0[%[[c0]], %[[c5]]]
// CHECK-DAG: %[[INS5:.*]] = tensor.insert %[[EXT5]] into %[[INS4]][%[[c0]], %[[c2]]]
// CHECK-DAG: %[[EXT6:.*]] = tensor.extract %arg0[%[[c0]], %[[c6]]]
// CHECK-DAG: %[[INS6:.*]] = tensor.insert %[[EXT6]] into %[[INS5]][%[[c0]], %[[c1]]]
// CHECK-DAG: %[[EXT7:.*]] = tensor.extract %arg0[%[[c0]], %[[c7]]]
// CHECK-DAG: %[[INS7:.*]] = tensor.insert %[[EXT7]] into %[[INS6]][%[[c0]], %[[c0]]]

// CHECK: @permutate_vector
#layout = dense<[[0, 0, 0, 7], [0, 1, 0, 6], [0, 2, 0, 5], [0, 3, 0, 4], [0, 4, 0, 3], [0, 5, 0, 2], [0, 6, 0, 1], [0, 7, 0, 0]]> : tensor<8x4xi64>
module {
  func.func @permutate_vector() {
    %cst = arith.constant dense<1> : tensor<1x32xi16>
    // CHECK: %[[cst:.*]] = arith.constant dense<1> : tensor<1x32xi16>
    // CHECK: func.call @_assign_layout_{{[0-9]+}}(%[[cst]])
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<1x32xi16>
      secret.yield %1 : tensor<1x32xi16>
    } -> (!secret.secret<tensor<1x32xi16>> {tensor_ext.layout = #layout})
    return
  }
}
