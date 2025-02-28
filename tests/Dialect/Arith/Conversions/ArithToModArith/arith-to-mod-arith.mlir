// RUN: heir-opt --arith-to-mod-arith --split-input-file %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: !Z2147483648_i33_, %[[RHS:.*]]: !Z2147483648_i33_) -> [[T:.*]] {
func.func @test_lower_add(%lhs : i32, %rhs : i32) -> i32 {
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.addi %lhs, %rhs : i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: tensor<4x!Z2147483648_i33_>, %[[RHS:.*]]: tensor<4x!Z2147483648_i33_>) -> [[T:.*]] {
func.func @test_lower_add_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[ADD:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.addi %lhs, %rhs : tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[ADD:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.subi %lhs, %rhs : tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : i32, %rhs : i32) -> i32 {
  // CHECK: %[[ADD:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.subi %lhs, %rhs : i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul(%lhs : i32, %rhs : i32) -> i32 {
  // CHECK: %[[ADD:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.muli %lhs, %rhs : i32
  return %res : i32
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: %[[ADD:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: return %[[ADD:.*]] : [[T]]
  %res = arith.muli %lhs, %rhs : tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK-LABEL: @test_arith_constant
// CHECK-SAME: () -> [[T:.*]] {
func.func @test_arith_constant() -> i32 {
  // CHECK: %[[C:.*]] = mod_arith.constant 17 : [[T]]
  // CHECK: return %[[C:.*]] : [[T]]
  %c17 = arith.constant 17 : i32
  return %c17 : i32
}

// CHECK-LABEL: @test_arith_constant_no_convert_index
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x[[T:.*]]>) -> [[T]] {
func.func @test_arith_constant_no_convert_index(%arg : tensor<2xi32>) -> i32 {
  // CHECK: %[[IDX:.*]] = arith.constant 17 : index
  %idx = arith.constant 17 : index
  // CHECK: %[[RES:.*]] = tensor.extract %[[ARG]][%[[IDX]]] : tensor<2x[[T]]>
  %res = tensor.extract %arg[%idx] : tensor<2xi32>
  // CHECK: return %[[RES:.*]] : [[T]]
  return %res : i32
}

// CHECK-LABEL: @test_memref_global
// CHECK-SAME: (%[[ARG:.*]]: memref<1x1x!Z2147483648_i33_>) -> memref<1x1x!Z2147483648_i33_> {
module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_16xi32_0 : memref<16xi32> = dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]> {alignment = 64 : i64}
  func.func @test_memref_global(%arg0: memref<1x1xi32>) -> memref<1x1xi32> {
    %c429_i32 = arith.constant 429 : i32
    %c33_i8 = arith.constant 33 : i8
    %c33 = arith.extui %c33_i8 : i8 to i32
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
    %3 = memref.get_global @__constant_16xi32_0 : memref<16xi32>
    %21 = memref.load %3[%c0] : memref<16xi32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
    %22 = memref.load %0[%c0, %c0] : memref<16x1xi8>
    %24 = memref.load %arg0[%c0, %c0] : memref<1x1xi32>
  // CHECK: %[[ENC:.*]] = mod_arith.mod_switch %{{.*}}: !Z128_i9_ to !Z2147483648_i33_
    %a24 = arith.extui %22 : i8 to i32
    %25 = arith.muli %24, %a24 : i32
    %26 = arith.addi %21, %25 : i32
    %27 = arith.addi %26, %c33 : i32
    memref.store %27, %alloc[%c0, %c0] : memref<1x1xi32>
    return %alloc : memref<1x1xi32>
  }
}

// CHECK-LABEL: @test_affine
// CHECK-SAME: (%[[ARG:.*]]: memref<1x1x!Z128_i9_>) -> memref<1x1x!Z2147483648_i33_> {
module attributes {tf_saved_model.semantics} {
  func.func @test_affine(%arg0: memref<1x1xi8>) -> memref<1x1xi32> {
    %c429_i32 = arith.constant 429 : i32
    %c33_i8 = arith.constant 33 : i8
    %c33 = arith.extui %c33_i8 : i8 to i32
    %0 = affine.load %arg0[0, 0] : memref<1x1xi8>
    %c0 = arith.constant 0 : index
    %1 = arith.extsi %0 : i8 to i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
  // CHECK: %[[ENC:.*]] = mod_arith.mod_switch %{{.*}}: !Z128_i9_ to !Z2147483648_i33_
    %25 = arith.muli %1, %c33 : i32
    %26 = arith.addi %c429_i32, %25 : i32
    affine.store %26, %alloc[0, 0] : memref<1x1xi32>
    return %alloc : memref<1x1xi32>
  }
}
