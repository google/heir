// RUN: heir-opt -yosys-optimizer="abc-fast=true" %s | FileCheck %s

// Regression test for https://github.com/google/heir/issues/359 When there are
// > 10 ports, the RTLIL wire ordering is not the same as the original generic's
// argument order.

module attributes {tf_saved_model.semantics} {
  // memref.global "private" constant @__constant_1x16xi8 : memref<1x16xi8> = dense<[[-39, 59, 39, 21, 28, -32, -34, -35, 15, 27, -59, -41, 18, -35, -7, 127]]>
  // memref.global "private" constant @__constant_16xi32_0 : memref<16xi32> = dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]>
  // memref.global "private" constant @__constant_16x16xi8 : memref<16x16xi8> = dense<"0xF41AED091921F424E021EFBCF7F5FA1903DCD20206F9F402FFFAEFF1EFD327E1FB27DDEBDBE4051A17FC241215EF1EE410FE14DA1CF8F3F1EFE2F309E3E9EDE3E415070B041B1AFEEB01DE21E60BEC03230A22241E2703E60324FFC011F8FCF1110CF5E0F30717E5E8EDFADCE823FB07DDFBFD0014261117E7F111EA0226040425211D0ADB1DDC2001FAE3370BF11A16EF1CE703E01602032118092ED9E5140BEA1AFCD81300C4D8ECD9FE0D1920D8D6E21FE9D7CAE2DDC613E7043E000114C7DBE71515F506D61ADC0922FE080213EF191EE209FDF314DDDA20D90FE3F9F7EEE924E629000716E21E0D23D3DDF714FA0822262109080F0BE012F47FDC58E526">
  // memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]>
  // memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]>
  // CHECK: @main
  // The only arith op we expect is arith.constant
  // CHECK-NOT: arith.{{^constant}}
  // CHECK: comb.truth_table
  func.func @main(%arg0: !secret.secret<tensor<1x1xi16>>) -> (!secret.secret<tensor<1x1xi8>>) {

    %__constant_1x16xi8 = arith.constant dense<[[-39, 59, 39, 21, 28, -32, -34, -35, 15, 27, -59, -41, 18, -35, -7, 127]]> : tensor<1x16xi8>
    %__constant_16xi32_0 = arith.constant dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]> : tensor<16xi32>
    %__constant_16x16xi8 = arith.constant dense<"0xF41AED091921F424E021EFBCF7F5FA1903DCD20206F9F402FFFAEFF1EFD327E1FB27DDEBDBE4051A17FC241215EF1EE410FE14DA1CF8F3F1EFE2F309E3E9EDE3E415070B041B1AFEEB01DE21E60BEC03230A22241E2703E60324FFC011F8FCF1110CF5E0F30717E5E8EDFADCE823FB07DDFBFD0014261117E7F111EA0226040425211D0ADB1DDC2001FAE3370BF11A16EF1CE703E01602032118092ED9E5140BEA1AFCD81300C4D8ECD9FE0D1920D8D6E21FE9D7CAE2DDC613E7043E000114C7DBE71515F506D61ADC0922FE080213EF191EE209FDF314DDDA20D90FE3F9F7EEE924E629000716E21E0D23D3DDF714FA0822262109080F0BE012F47FDC58E526"> : tensor<16x16xi8>
    %__constant_16xi32 = arith.constant dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]> : tensor<16xi32>
    %__constant_16x1xi8 = arith.constant dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]> : tensor<16x1xi8>

    %c34359738368_i64 = arith.constant 3435973 : i32
    %c1630361836_i64 = arith.constant 1630361836 : i32
    %c36_i64 = arith.constant 36 : i32
    %c5_i32 = arith.constant 5 : i16
    %c127_i32 = arith.constant 127 : i16
    %c-1073741824_i64 = arith.constant -1073741824 : i32
    %c1073741824_i64 = arith.constant 1073741824 : i32
    %c0_i32 = arith.constant 0 : i16
    %c-128_i32 = arith.constant -128 : i16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %init_tensor = secret.generic() {
      %alloc = tensor.empty() {alignment = 64 : i64} : tensor<1x1xi8>
      secret.yield %alloc : tensor<1x1xi8>
    } -> !secret.secret<tensor<1x1xi8>>

    %result_tensor = affine.for  %arg1 = 0 to 1 iter_args(%iter_tensor_outer = %init_tensor) -> (!secret.secret<tensor<1x1xi8>>) {
      %inner_loop_res = affine.for %arg2 = 0 to 1 iter_args(%iter_tensor_inner = %iter_tensor_outer) -> (!secret.secret<tensor<1x1xi8>>) {
        // 3. Extract a value from the input tensor instead of loading from memref.
        %1 = secret.generic(%arg0: !secret.secret<tensor<1x1xi16>>, %arg1: index, %arg2: index) {
        ^bb0(%arg3: tensor<1x1xi16>, %arg4: index, %arg5: index):
          %3 = tensor.extract %arg3[%arg4, %arg5] : tensor<1x1xi16>
          secret.yield %3 : i16
        } -> !secret.secret<i16>

        %2 = secret.generic(%1: !secret.secret<i16>, %c1630361836_i64: i32, %c34359738368_i64: i32, %c0_i32: i16, %c1073741824_i64: i32, %c-1073741824_i64: i32, %c36_i64: i32, %c5_i32: i16, %c-128_i32: i16, %c127_i32: i16) {
        ^bb0(%arg3: i16, %arg4: i32, %arg5: i32, %arg6: i16, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i16, %arg11: i16, %arg12: i16):
          %3 = arith.extsi %arg3 : i16 to i32
          %4 = arith.muli %3, %arg4 : i32
          %5 = arith.addi %4, %arg5 : i32
          %6 = arith.cmpi sge, %arg3, %arg6 : i16
          %7 = arith.select %6, %arg7, %arg8 : i32
          %8 = arith.addi %7, %5 : i32
          %9 = arith.shrsi %8, %arg9 : i32
          %10 = arith.trunci %9 : i32 to i16
          %11 = arith.addi %10, %arg10 : i16
          %12 = arith.cmpi slt, %11, %arg11 : i16
          %13 = arith.select %12, %arg11, %11 : i16
          %14 = arith.cmpi sgt, %11, %arg12 : i16
          %15 = arith.select %14, %arg12, %13 : i16
          %16 = arith.trunci %15 : i16 to i8
          secret.yield %16 : i8
        } -> !secret.secret<i8>

        %updated_tensor = secret.generic(%iter_tensor_inner: !secret.secret<tensor<1x1xi8>>, %2: !secret.secret<i8>, %arg1: index, %arg2: index) {
        ^bb0(%arg3: tensor<1x1xi8>, %arg4: i8, %arg5: index, %arg6: index):
          %new_tensor = tensor.insert %arg4 into %arg3[%arg5, %arg6] : tensor<1x1xi8>
          secret.yield %new_tensor : tensor<1x1xi8>
        } -> !secret.secret<tensor<1x1xi8>>

        affine.yield %updated_tensor : !secret.secret<tensor<1x1xi8>>
      }
      affine.yield %inner_loop_res : !secret.secret<tensor<1x1xi8>>
    }
    // CHECK: return
    return %result_tensor : !secret.secret<tensor<1x1xi8>>
  }
}
