// // RUN: heir-opt --arith-to-mod-arith --split-input-file %s | FileCheck %s --enable-var-scope

// // CHECK-LABEL: @test_lower_add
// // CHECK-SAME: (%[[LHS:.*]]: !Z2147483647_i32_, %[[RHS:.*]]: !Z2147483647_i32_) -> [[T:.*]] {
// func.func @test_lower_add(%lhs : i32, %rhs : i32) -> i32 {
//   // CHECK: %[[ADD:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : [[T]]
//   // CHECK: return %[[ADD:.*]] : [[T]]
//   %res = arith.addi %lhs, %rhs : i32
//   return %res : i32
// }

// // CHECK-LABEL: @test_lower_add_vec
// // CHECK-SAME: (%[[LHS:.*]]: tensor<4x!Z2147483647_i32_>, %[[RHS:.*]]: tensor<4x!Z2147483647_i32_>) -> [[T:.*]] {
// func.func @test_lower_add_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
//   // CHECK: %[[ADD:.*]] = mod_arith.add %[[LHS]], %[[RHS]] : [[T]]
//   // CHECK: return %[[ADD:.*]] : [[T]]
//   %res = arith.addi %lhs, %rhs : tensor<4xi32>
//   return %res : tensor<4xi32>
// }

// // CHECK-LABEL: @test_lower_sub_vec
// // CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
// func.func @test_lower_sub_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
//   // CHECK: %[[ADD:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : [[T]]
//   // CHECK: return %[[ADD:.*]] : [[T]]
//   %res = arith.subi %lhs, %rhs : tensor<4xi32>
//   return %res : tensor<4xi32>
// }

// // CHECK-LABEL: @test_lower_sub
// // CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
// func.func @test_lower_sub(%lhs : i32, %rhs : i32) -> i32 {
//   // CHECK: %[[ADD:.*]] = mod_arith.sub %[[LHS]], %[[RHS]] : [[T]]
//   // CHECK: return %[[ADD:.*]] : [[T]]
//   %res = arith.subi %lhs, %rhs : i32
//   return %res : i32
// }

// // CHECK-LABEL: @test_lower_mul
// // CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
// func.func @test_lower_mul(%lhs : i32, %rhs : i32) -> i32 {
//   // CHECK: %[[ADD:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : [[T]]
//   // CHECK: return %[[ADD:.*]] : [[T]]
//   %res = arith.muli %lhs, %rhs : i32
//   return %res : i32
// }

// // CHECK-LABEL: @test_lower_mul_vec
// // CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
// func.func @test_lower_mul_vec(%lhs : tensor<4xi32>, %rhs : tensor<4xi32>) -> tensor<4xi32> {
//   // CHECK: %[[ADD:.*]] = mod_arith.mul %[[LHS]], %[[RHS]] : [[T]]
//   // CHECK: return %[[ADD:.*]] : [[T]]
//   %res = arith.muli %lhs, %rhs : tensor<4xi32>
//   return %res : tensor<4xi32>
// }

// module {
//   memref.global "private" constant @__constant_1x16xi8 : memref<1x16xi32> = dense<[[39, 59, 39, 21, 28, 2, 34, 35, 15, 27, 59, 41, 18, 35, 7, 127]]> {alignment = 64 : i64}
// func.func @test_mac(%lhs : i32, %rhs : i32) -> i32 {
//     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
//   %c0 = arith.constant 0 : index
//   %c6 = arith.constant 6 : index
//   %c = arith.constant 32: i32
//   %4 = memref.get_global @__constant_1x16xi8 : memref<1x16xi32>
//   %661 = memref.load %4[%c0, %c6] : memref<1x16xi32>
//   // %27 = arith.extsi %661 : i8 to i32
//   %25 = arith.muli %lhs, %rhs : i32
//   %26 = arith.addi %c, %25 : i32
//   %res = arith.muli %26, %661 : i32
//   return %res : i32
// }
// }

module attributes {tf_saved_model.semantics} {
  // memref.global "private" constant @__constant_1x16xi8 : memref<1x16xi8> = dense<[[-39, 59, 39, 21, 28, -32, -34, -35, 15, 27, -59, -41, 18, -35, -7, 127]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16xi32_0 : memref<16xi32> = dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]> {alignment = 64 : i64}
  // memref.global "private" constant @__constant_16x16xi8 : memref<16x16xi8> = dense<"0xF41AED091921F424E021EFBCF7F5FA1903DCD20206F9F402FFFAEFF1EFD327E1FB27DDEBDBE4051A17FC241215EF1EE410FE14DA1CF8F3F1EFE2F309E3E9EDE3E415070B041B1AFEEB01DE21E60BEC03230A22241E2703E60324FFC011F8FCF1110CF5E0F30717E5E8EDFADCE823FB07DDFBFD0014261117E7F111EA0226040425211D0ADB1DDC2001FAE3370BF11A16EF1CE703E01602032118092ED9E5140BEA1AFCD81300C4D8ECD9FE0D1920D8D6E21FE9D7CAE2DDC613E7043E000114C7DBE71515F506D61ADC0922FE080213EF191EE209FDF314DDDA20D90FE3F9F7EEE924E629000716E21E0D23D3DDF714FA0822262109080F0BE012F47FDC58E526"> {alignment = 64 : i64}
  // memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]> {alignment = 64 : i64}
  func.func @main(%arg0: memref<1x1xi8>) -> i32 {
    %c429_i32 = arith.constant 429 : i32
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
    // %1 = memref.get_global @__constant_16xi32 : memref<16xi32>
    // %2 = memref.get_global @__constant_16x16xi8 : memref<16x16xi8>
    // %3 = memref.get_global @__constant_16xi32_0 : memref<16xi32>
    // %4 = memref.get_global @__constant_1x16xi8 : memref<1x16xi8>
    // %21 = memref.load %3[%c0] : memref<16xi32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
    %22 = memref.load %0[%c0, %c0] : memref<16x1xi8>
    %23 = memref.load %arg0[%c0, %c0] : memref<1x1xi8>
    %24 = arith.extsi %23 : i8 to i32
        %a24 = arith.extsi %22 : i8 to i32
    %25 = arith.muli %24, %a24 : i32
    %26 = arith.addi %25, %25 : i32
    %29 = arith.muli %26, %25 : i32
    %30 = arith.addi %26, %29 : i32
    return %30 : i32
  }
  }


// Problem is: memref cannot be used as return type (allloc is okey)
// Also the input is a problem
