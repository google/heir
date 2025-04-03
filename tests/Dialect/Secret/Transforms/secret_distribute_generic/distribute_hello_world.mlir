// RUN: heir-opt --secret-distribute-generic=distribute-through="affine.for,affine.load,affine.store,memref.get_global" %s | FileCheck %s

module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_1x16xi8 : memref<1x16xi8> = dense<[[-39, 59, 39, 21, 28, -32, -34, -35, 15, 27, -59, -41, 18, -35, -7, 127]]>
  memref.global "private" constant @__constant_16xi32_0 : memref<16xi32> = dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]>
  memref.global "private" constant @__constant_16x16xi8 : memref<16x16xi8> = dense<"0xF41AED091921F424E021EFBCF7F5FA1903DCD20206F9F402FFFAEFF1EFD327E1FB27DDEBDBE4051A17FC241215EF1EE410FE14DA1CF8F3F1EFE2F309E3E9EDE3E415070B041B1AFEEB01DE21E60BEC03230A22241E2703E60324FFC011F8FCF1110CF5E0F30717E5E8EDFADCE823FB07DDFBFD0014261117E7F111EA0226040425211D0ADB1DDC2001FAE3370BF11A16EF1CE703E01602032118092ED9E5140BEA1AFCD81300C4D8ECD9FE0D1920D8D6E21FE9D7CAE2DDC613E7043E000114C7DBE71515F506D61ADC0922FE080213EF191EE209FDF314DDDA20D90FE3F9F7EEE924E629000716E21E0D23D3DDF714FA0822262109080F0BE012F47FDC58E526">
  memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]>
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]>
  // CHECK: main
  // CHECK-SAME: %[[value:.*]]: !secret.secret<memref<1x1xi8>>
  func.func @main(%arg0: !secret.secret<memref<1x1xi8>> {iree.identifier = "serving_default_dense_input:0", tf_saved_model.index_path = ["dense_input"]}) -> (!secret.secret<memref<1x1xi8>> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    // CHECK-NOT: secret.generic
    %0 = secret.generic ins(%arg0 : !secret.secret<memref<1x1xi8>>) {
    ^body(%arg1: memref<1x1xi8>):
      %c-128_i32 = arith.constant -128 : i32
      %c0_i32 = arith.constant 0 : i32
      %c1073741824_i64 = arith.constant 1073741824 : i64
      %c-1073741824_i64 = arith.constant -1073741824 : i64
      %c127_i32 = arith.constant 127 : i32
      %c-128_i8 = arith.constant -128 : i8
      %c127_i8 = arith.constant 127 : i8
      %c5_i32 = arith.constant 5 : i32
      %c36_i64 = arith.constant 36 : i64
      %c1630361836_i64 = arith.constant 1630361836 : i64
      %c34359738368_i64 = arith.constant 34359738368 : i64
      %c37_i64 = arith.constant 37 : i64
      %c1561796795_i64 = arith.constant 1561796795 : i64
      %c68719476736_i64 = arith.constant 68719476736 : i64
      %c38_i64 = arith.constant 38 : i64
      %c137438953472_i64 = arith.constant 137438953472 : i64
      %c2039655736_i64 = arith.constant 2039655736 : i64
      %c429_i32 = arith.constant 429 : i32
      %c-128_i16 = arith.constant -128 : i16
    // CHECK-COUNT-5: memref.get_global
      %1 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
      %2 = memref.get_global @__constant_16xi32 : memref<16xi32>
      %3 = memref.get_global @__constant_16x16xi8 : memref<16x16xi8>
      %4 = memref.get_global @__constant_16xi32_0 : memref<16xi32>
      %5 = memref.get_global @__constant_1x16xi8 : memref<1x16xi8>
    // CHECK: secret.generic
    // CHECK-NEXT: memref.alloc
    // CHECK-NEXT: secret.yield
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      // CHECK: affine.for
      affine.for %arg2 = 0 to 1 {
      // CHECK: affine.for
        affine.for %arg3 = 0 to 16 {
            // CHECK: affine.load
          %7 = affine.load %1[%arg3, %arg2] : memref<16x1xi8>
            // CHECK: secret.generic
            // CHECK-NEXT: ^body
            // CHECK-NEXT: affine.store
            // CHECK-NEXT: secret.yield
          affine.store %7, %alloc[%arg2, %arg3] : memref<1x16xi8>
        }
      }
    // CHECK: secret.generic
    // CHECK-NEXT: memref.alloc
    // CHECK-NEXT: secret.yield
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      // CHECK: affine.for
      affine.for %arg2 = 0 to 1 {
      // CHECK: affine.for
        affine.for %arg3 = 0 to 16 {
          %7 = affine.load %2[%arg3] : memref<16xi32>
          affine.store %7, %alloc_0[%arg2, %arg3] : memref<1x16xi32>
        }
      }
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      // CHECK: affine.for
      affine.for %arg2 = 0 to 16 {
        %7 = affine.load %alloc_0[0, %arg2] : memref<1x16xi32>
        affine.store %7, %alloc_1[0, %arg2] : memref<1x16xi32>
      }
      // CHECK: affine.for
      affine.for %arg2 = 0 to 1 {
      // CHECK: affine.for
        affine.for %arg3 = 0 to 16 {
      // CHECK: affine.for
          affine.for %arg4 = 0 to 1 {
            // CHECK-COUNT-3: secret.generic
            %7 = affine.load %arg1[%arg2, %arg4] : memref<1x1xi8>
            %8 = affine.load %alloc[%arg4, %arg3] : memref<1x16xi8>
            %9 = affine.load %alloc_1[%arg2, %arg3] : memref<1x16xi32>
            // CHECK: secret.generic
            // CHECK-NEXT:   ^body
            // CHECK-NEXT: arith.extsi
            // CHECK-NEXT: arith.subi
            // CHECK-NEXT: arith.extsi
            // CHECK-NEXT: arith.extsi
            // CHECk-NEXT: arith.muli
            %10 = arith.extsi %7 : i8 to i16
            %11 = arith.subi %10, %c-128_i16 : i16
            %12 = arith.extsi %11 : i16 to i32
            %13 = arith.extsi %8 : i8 to i32
            %14 = arith.muli %12, %13 : i32
            %15 = arith.addi %9, %14 : i32
            // CHECK: secret.generic
            // CHECK-NEXT:   ^body
            // CHECK-NEXT: affine.store
            // CHECK-NEXT: secret.yield
            affine.store %15, %alloc_1[%arg2, %arg3] : memref<1x16xi32>
          }
        }
      }
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          %7 = affine.load %alloc_1[%arg2, %arg3] : memref<1x16xi32>
          %8 = arith.extsi %7 : i32 to i64
          %9 = arith.muli %8, %c2039655736_i64 : i64
          %10 = arith.addi %9, %c137438953472_i64 : i64
          %11 = arith.cmpi sge, %7, %c0_i32 : i32
          %12 = arith.select %11, %c1073741824_i64, %c-1073741824_i64 : i64
          %13 = arith.addi %12, %10 : i64
          %14 = arith.shrsi %13, %c38_i64 : i64
          %15 = arith.trunci %14 : i64 to i32
          %16 = arith.addi %15, %c-128_i32 : i32
          %17 = arith.cmpi slt, %16, %c-128_i32 : i32
          %18 = arith.select %17, %c-128_i32, %16 : i32
          %19 = arith.cmpi sgt, %16, %c127_i32 : i32
          %20 = arith.select %19, %c127_i32, %18 : i32
          %21 = arith.trunci %20 : i32 to i8
          affine.store %21, %alloc_2[%arg2, %arg3] : memref<1x16xi8>
        }
      }
      %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          %7 = affine.load %alloc_2[0, %arg3] : memref<1x16xi8>
          %8 = arith.cmpi slt, %7, %c-128_i8 : i8
          %9 = arith.select %8, %c-128_i8, %7 : i8
          %10 = arith.cmpi sgt, %7, %c127_i8 : i8
          %11 = arith.select %10, %c127_i8, %9 : i8
          affine.store %11, %alloc_3[%arg2, %arg3] : memref<1x16xi8>
        }
      }
      %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
      affine.for %arg2 = 0 to 16 {
        affine.for %arg3 = 0 to 16 {
          %7 = affine.load %3[%arg3, %arg2] : memref<16x16xi8>
          affine.store %7, %alloc_4[%arg2, %arg3] : memref<16x16xi8>
        }
      }
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          %7 = affine.load %4[%arg3] : memref<16xi32>
          affine.store %7, %alloc_5[%arg2, %arg3] : memref<1x16xi32>
        }
      }
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      affine.for %arg2 = 0 to 16 {
        %7 = affine.load %alloc_5[0, %arg2] : memref<1x16xi32>
        affine.store %7, %alloc_6[0, %arg2] : memref<1x16xi32>
      }
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          affine.for %arg4 = 0 to 16 {
            %7 = affine.load %alloc_3[%arg2, %arg4] : memref<1x16xi8>
            %8 = affine.load %alloc_4[%arg4, %arg3] : memref<16x16xi8>
            %9 = affine.load %alloc_6[%arg2, %arg3] : memref<1x16xi32>
            %10 = arith.extsi %7 : i8 to i16
            %11 = arith.subi %10, %c-128_i16 : i16
            %12 = arith.extsi %11 : i16 to i32
            %13 = arith.extsi %8 : i8 to i32
            %14 = arith.muli %12, %13 : i32
            %15 = arith.addi %9, %14 : i32
            affine.store %15, %alloc_6[%arg2, %arg3] : memref<1x16xi32>
          }
        }
      }
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          %7 = affine.load %alloc_6[%arg2, %arg3] : memref<1x16xi32>
          %8 = arith.extsi %7 : i32 to i64
          %9 = arith.muli %8, %c1561796795_i64 : i64
          %10 = arith.addi %9, %c68719476736_i64 : i64
          %11 = arith.cmpi sge, %7, %c0_i32 : i32
          %12 = arith.select %11, %c1073741824_i64, %c-1073741824_i64 : i64
          %13 = arith.addi %12, %10 : i64
          %14 = arith.shrsi %13, %c37_i64 : i64
          %15 = arith.trunci %14 : i64 to i32
          %16 = arith.addi %15, %c-128_i32 : i32
          %17 = arith.cmpi slt, %16, %c-128_i32 : i32
          %18 = arith.select %17, %c-128_i32, %16 : i32
          %19 = arith.cmpi sgt, %16, %c127_i32 : i32
          %20 = arith.select %19, %c127_i32, %18 : i32
          %21 = arith.trunci %20 : i32 to i8
          affine.store %21, %alloc_7[%arg2, %arg3] : memref<1x16xi8>
        }
      }
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          %7 = affine.load %alloc_7[0, %arg3] : memref<1x16xi8>
          %8 = arith.cmpi slt, %7, %c-128_i8 : i8
          %9 = arith.select %8, %c-128_i8, %7 : i8
          %10 = arith.cmpi sgt, %7, %c127_i8 : i8
          %11 = arith.select %10, %c127_i8, %9 : i8
          affine.store %11, %alloc_8[%arg2, %arg3] : memref<1x16xi8>
        }
      }
      %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<16x1xi8>
      affine.for %arg2 = 0 to 16 {
        affine.for %arg3 = 0 to 1 {
          %7 = affine.load %5[%arg3, %arg2] : memref<1x16xi8>
          affine.store %7, %alloc_9[%arg2, %arg3] : memref<16x1xi8>
        }
      }
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.store %c429_i32, %alloc_10[%arg2, %arg3] : memref<1x1xi32>
        }
      }
      %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
      %6 = affine.load %alloc_10[0, 0] : memref<1x1xi32>
      affine.store %6, %alloc_11[0, 0] : memref<1x1xi32>
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 16 {
            %7 = affine.load %alloc_8[%arg2, %arg4] : memref<1x16xi8>
            %8 = affine.load %alloc_9[%arg4, %arg3] : memref<16x1xi8>
            %9 = affine.load %alloc_11[%arg2, %arg3] : memref<1x1xi32>
            %10 = arith.extsi %7 : i8 to i16
            %11 = arith.subi %10, %c-128_i16 : i16
            %12 = arith.extsi %11 : i16 to i32
            %13 = arith.extsi %8 : i8 to i32
            %14 = arith.muli %12, %13 : i32
            %15 = arith.addi %9, %14 : i32
            affine.store %15, %alloc_11[%arg2, %arg3] : memref<1x1xi32>
          }
        }
      }
      %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi8>
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1 {
          %7 = affine.load %alloc_11[%arg2, %arg3] : memref<1x1xi32>
          %8 = arith.extsi %7 : i32 to i64
          %9 = arith.muli %8, %c1630361836_i64 : i64
          %10 = arith.addi %9, %c34359738368_i64 : i64
          %11 = arith.cmpi sge, %7, %c0_i32 : i32
          %12 = arith.select %11, %c1073741824_i64, %c-1073741824_i64 : i64
          %13 = arith.addi %12, %10 : i64
          %14 = arith.shrsi %13, %c36_i64 : i64
          %15 = arith.trunci %14 : i64 to i32
          %16 = arith.addi %15, %c5_i32 : i32
          %17 = arith.cmpi slt, %16, %c-128_i32 : i32
          %18 = arith.select %17, %c-128_i32, %16 : i32
          %19 = arith.cmpi sgt, %16, %c127_i32 : i32
          %20 = arith.select %19, %c127_i32, %18 : i32
          %21 = arith.trunci %20 : i32 to i8
          affine.store %21, %alloc_12[%arg2, %arg3] : memref<1x1xi8>
        }
      }
      secret.yield %alloc_12 : memref<1x1xi8>
    } -> !secret.secret<memref<1x1xi8>>
    return %0 : !secret.secret<memref<1x1xi8>>
  }
}
