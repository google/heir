// RUN: heir-opt --canonicalize %s | FileCheck %s

module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_1x16xi8 : memref<1x16xi8> = dense<[[-39, 59, 39, 21, 28, -32, -34, -35, 15, 27, -59, -41, 18, -35, -7, 127]]>
  memref.global "private" constant @__constant_16xi32_0 : memref<16xi32> = dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]>
  memref.global "private" constant @__constant_16x16xi8 : memref<16x16xi8> = dense<"0xF41AED091921F424E021EFBCF7F5FA1903DCD20206F9F402FFFAEFF1EFD327E1FB27DDEBDBE4051A17FC241215EF1EE410FE14DA1CF8F3F1EFE2F309E3E9EDE3E415070B041B1AFEEB01DE21E60BEC03230A22241E2703E60324FFC011F8FCF1110CF5E0F30717E5E8EDFADCE823FB07DDFBFD0014261117E7F111EA0226040425211D0ADB1DDC2001FAE3370BF11A16EF1CE703E01602032118092ED9E5140BEA1AFCD81300C4D8ECD9FE0D1920D8D6E21FE9D7CAE2DDC613E7043E000114C7DBE71515F506D61ADC0922FE080213EF191EE209FDF314DDDA20D90FE3F9F7EEE924E629000716E21E0D23D3DDF714FA0822262109080F0BE012F47FDC58E526">
  memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]>
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]>
// CHECK: func @main
  func.func @main(%arg0: !secret.secret<memref<1x1xi8>> {iree.identifier = "serving_default_dense_input:0", tf_saved_model.index_path = ["dense_input"]}) -> (!secret.secret<memref<1x16xi32>> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %c-128_i16 = arith.constant -128 : i16
    %c429_i32 = arith.constant 429 : i32
    %c2039655736_i64 = arith.constant 2039655736 : i64
    %c137438953472_i64 = arith.constant 137438953472 : i64
    %c38_i64 = arith.constant 38 : i64
    %c68719476736_i64 = arith.constant 68719476736 : i64
    %c1561796795_i64 = arith.constant 1561796795 : i64
    %c37_i64 = arith.constant 37 : i64
    %c34359738368_i64 = arith.constant 34359738368 : i64
    %c1630361836_i64 = arith.constant 1630361836 : i64
    %c36_i64 = arith.constant 36 : i64
    %c5_i32 = arith.constant 5 : i32
    %c127_i8 = arith.constant 127 : i8
    %c-128_i8 = arith.constant -128 : i8
    %c127_i32 = arith.constant 127 : i32
    %c-1073741824_i64 = arith.constant -1073741824 : i64
    %c1073741824_i64 = arith.constant 1073741824 : i64
    %c0_i32 = arith.constant 0 : i32
    %c-128_i32 = arith.constant -128 : i32
    %0 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
    %1 = memref.get_global @__constant_16xi32 : memref<16xi32>
    %2 = memref.get_global @__constant_16x16xi8 : memref<16x16xi8>
    %3 = memref.get_global @__constant_16xi32_0 : memref<16xi32>
    %4 = memref.get_global @__constant_1x16xi8 : memref<1x16xi8>
    %5 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      secret.yield %alloc : memref<1x16xi8>
    } -> !secret.secret<memref<1x16xi8>>
    // CHECK-COUNT-5: affine.for
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %20 = affine.load %0[%arg2, %arg1] : memref<16x1xi8>
        secret.generic ins(%5 : !secret.secret<memref<1x16xi8>>) {
        ^bb0(%arg3: memref<1x16xi8>):
          affine.store %20, %arg3[%arg1, %arg2] : memref<1x16xi8>
          secret.yield
        }
      }
    }
    %6 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      secret.yield %alloc : memref<1x16xi32>
    } -> !secret.secret<memref<1x16xi32>>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %20 = affine.load %1[%arg2] : memref<16xi32>
        secret.generic ins(%6 : !secret.secret<memref<1x16xi32>>) {
        ^bb0(%arg3: memref<1x16xi32>):
          affine.store %20, %arg3[%arg1, %arg2] : memref<1x16xi32>
          secret.yield
        }
      }
    }
    %7 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      secret.yield %alloc : memref<1x16xi32>
    } -> !secret.secret<memref<1x16xi32>>
    affine.for %arg1 = 0 to 16 {
      %20 = secret.generic ins(%6 : !secret.secret<memref<1x16xi32>>) {
      ^bb0(%arg2: memref<1x16xi32>):
        %21 = affine.load %arg2[0, %arg1] : memref<1x16xi32>
        secret.yield %21 : i32
      } -> !secret.secret<i32>
      secret.generic ins(%7, %20 : !secret.secret<memref<1x16xi32>>, !secret.secret<i32>) {
      ^bb0(%arg2: memref<1x16xi32>, %arg3: i32):
        affine.store %arg3, %arg2[0, %arg1] : memref<1x16xi32>
        secret.yield
      }
    }
    // CHECK: affine.for
    affine.for %arg1 = 0 to 1 {
    // CHECK-NEXT: affine.for
      affine.for %arg2 = 0 to 16 {
    // CHECK-NEXT: affine.for
        affine.for %arg3 = 0 to 1 {
          // CHECK-COUNT-3: secret.generic
          %20 = secret.generic ins(%arg0 : !secret.secret<memref<1x1xi8>>) {
          ^bb0(%arg4: memref<1x1xi8>):
            %24 = affine.load %arg4[%arg1, %arg3] : memref<1x1xi8>
            secret.yield %24 : i8
          } -> !secret.secret<i8>
          %21 = secret.generic ins(%5 : !secret.secret<memref<1x16xi8>>) {
          ^bb0(%arg4: memref<1x16xi8>):
            %24 = affine.load %arg4[%arg3, %arg2] : memref<1x16xi8>
            secret.yield %24 : i8
          } -> !secret.secret<i8>
          %22 = secret.generic ins(%7 : !secret.secret<memref<1x16xi32>>) {
          ^bb0(%arg4: memref<1x16xi32>):
            %24 = affine.load %arg4[%arg1, %arg2] : memref<1x16xi32>
            secret.yield %24 : i32
          } -> !secret.secret<i32>
          // CHECK: secret.generic
          // CHECK: secret.yield
          // CHECK-NEXT: -> !secret.secret<i32>
          %23:6 = secret.generic ins(%20, %21, %22 : !secret.secret<i8>, !secret.secret<i8>, !secret.secret<i32>) {
          ^bb0(%arg4: i8, %arg5: i8, %arg6: i32):
            %24 = arith.extsi %arg4 : i8 to i16
            %25 = arith.subi %24, %c-128_i16 : i16
            %26 = arith.extsi %25 : i16 to i32
            %27 = arith.extsi %arg5 : i8 to i32
            %28 = arith.muli %26, %27 : i32
            %29 = arith.addi %arg6, %28 : i32
            secret.yield %24, %25, %26, %27, %28, %29 : i16, i16, i32, i32, i32, i32
          } -> (!secret.secret<i16>, !secret.secret<i16>, !secret.secret<i32>, !secret.secret<i32>, !secret.secret<i32>, !secret.secret<i32>)
          secret.generic ins(%7, %23#5 : !secret.secret<memref<1x16xi32>>, !secret.secret<i32>) {
          ^bb0(%arg4: memref<1x16xi32>, %arg5: i32):
            affine.store %arg5, %arg4[%arg1, %arg2] : memref<1x16xi32>
            secret.yield
          }
        }
      }
    }
    return %7 : !secret.secret<memref<1x16xi32>>
  }
}
