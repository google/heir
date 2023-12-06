// RUN: heir-opt --wrap-generic %s | FileCheck %s

// CHECK: module
module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_1x16xi8 : memref<1x16xi8> = dense<[[-39, 59, 39, 21, 28, -32, -34, -35, 15, 27, -59, -41, 18, -35, -7, 127]]>
  memref.global "private" constant @__constant_16xi32_0 : memref<16xi32> = dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]>
  memref.global "private" constant @__constant_16x16xi8 : memref<16x16xi8> = dense<"0xF41AED091921F424E021EFBCF7F5FA1903DCD20206F9F402FFFAEFF1EFD327E1FB27DDEBDBE4051A17FC241215EF1EE410FE14DA1CF8F3F1EFE2F309E3E9EDE3E415070B041B1AFEEB01DE21E60BEC03230A22241E2703E60324FFC011F8FCF1110CF5E0F30717E5E8EDFADCE823FB07DDFBFD0014261117E7F111EA0226040425211D0ADB1DDC2001FAE3370BF11A16EF1CE703E01602032118092ED9E5140BEA1AFCD81300C4D8ECD9FE0D1920D8D6E21FE9D7CAE2DDC613E7043E000114C7DBE71515F506D61ADC0922FE080213EF191EE209FDF314DDDA20D90FE3F9F7EEE924E629000716E21E0D23D3DDF714FA0822262109080F0BE012F47FDC58E526">
  memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]>
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]>
    // CHECK: @main(%[[ARG0:.*]]: !secret.secret<memref<1x1xi8>>
  func.func @main(%arg0: memref<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> (memref<1x1xi8> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    // CHECK: %[[V0:.*]] = secret.generic ins(%[[ARG0]] : !secret.secret<memref<1x1xi8>>)
    // CHECK: (%[[ARG1:.*]]: memref<1x1xi8>):
    // CHECK-NOT: [[ARG0]]
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
    %0 = memref.get_global @__constant_16x1xi8 : memref<16x1xi8>
    %1 = memref.get_global @__constant_16xi32 : memref<16xi32>
    %2 = memref.get_global @__constant_16x16xi8 : memref<16x16xi8>
    %3 = memref.get_global @__constant_16xi32_0 : memref<16xi32>
    %4 = memref.get_global @__constant_1x16xi8 : memref<1x16xi8>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %6 = affine.load %0[%arg2, %arg1] : memref<16x1xi8>
        affine.store %6, %alloc[%arg1, %arg2] : memref<1x16xi8>
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %6 = affine.load %1[%arg2] : memref<16xi32>
        affine.store %6, %alloc_0[%arg1, %arg2] : memref<1x16xi32>
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
    affine.for %arg1 = 0 to 16 {
      %6 = affine.load %alloc_0[0, %arg1] : memref<1x16xi32>
      affine.store %6, %alloc_1[0, %arg1] : memref<1x16xi32>
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        affine.for %arg3 = 0 to 1 {
          %6 = affine.load %arg0[%arg1, %arg3] : memref<1x1xi8>
          %7 = affine.load %alloc[%arg3, %arg2] : memref<1x16xi8>
          %8 = affine.load %alloc_1[%arg1, %arg2] : memref<1x16xi32>
          %9 = arith.extsi %6 : i8 to i16
          %10 = arith.subi %9, %c-128_i16 : i16
          %11 = arith.extsi %10 : i16 to i32
          %12 = arith.extsi %7 : i8 to i32
          %13 = arith.muli %11, %12 : i32
          %14 = arith.addi %8, %13 : i32
          affine.store %14, %alloc_1[%arg1, %arg2] : memref<1x16xi32>
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %6 = affine.load %alloc_1[%arg1, %arg2] : memref<1x16xi32>
        %7 = arith.extsi %6 : i32 to i64
        %8 = arith.muli %7, %c2039655736_i64 : i64
        %9 = arith.addi %8, %c137438953472_i64 : i64
        %10 = arith.cmpi sge, %6, %c0_i32 : i32
        %11 = arith.select %10, %c1073741824_i64, %c-1073741824_i64 : i64
        %12 = arith.addi %11, %9 : i64
        %13 = arith.shrsi %12, %c38_i64 : i64
        %14 = arith.trunci %13 : i64 to i32
        %15 = arith.addi %14, %c-128_i32 : i32
        %16 = arith.cmpi slt, %15, %c-128_i32 : i32
        %17 = arith.select %16, %c-128_i32, %15 : i32
        %18 = arith.cmpi sgt, %15, %c127_i32 : i32
        %19 = arith.select %18, %c127_i32, %17 : i32
        %20 = arith.trunci %19 : i32 to i8
        affine.store %20, %alloc_2[%arg1, %arg2] : memref<1x16xi8>
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %6 = affine.load %alloc_2[0, %arg2] : memref<1x16xi8>
        %7 = arith.cmpi slt, %6, %c-128_i8 : i8
        %8 = arith.select %7, %c-128_i8, %6 : i8
        %9 = arith.cmpi sgt, %6, %c127_i8 : i8
        %10 = arith.select %9, %c127_i8, %8 : i8
        affine.store %10, %alloc_3[%arg1, %arg2] : memref<1x16xi8>
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
    affine.for %arg1 = 0 to 16 {
      affine.for %arg2 = 0 to 16 {
        %6 = affine.load %2[%arg2, %arg1] : memref<16x16xi8>
        affine.store %6, %alloc_4[%arg1, %arg2] : memref<16x16xi8>
      }
    }
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %6 = affine.load %3[%arg2] : memref<16xi32>
        affine.store %6, %alloc_5[%arg1, %arg2] : memref<1x16xi32>
      }
    }
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
    affine.for %arg1 = 0 to 16 {
      %6 = affine.load %alloc_5[0, %arg1] : memref<1x16xi32>
      affine.store %6, %alloc_6[0, %arg1] : memref<1x16xi32>
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        affine.for %arg3 = 0 to 16 {
          %6 = affine.load %alloc_3[%arg1, %arg3] : memref<1x16xi8>
          %7 = affine.load %alloc_4[%arg3, %arg2] : memref<16x16xi8>
          %8 = affine.load %alloc_6[%arg1, %arg2] : memref<1x16xi32>
          %9 = arith.extsi %6 : i8 to i16
          %10 = arith.subi %9, %c-128_i16 : i16
          %11 = arith.extsi %10 : i16 to i32
          %12 = arith.extsi %7 : i8 to i32
          %13 = arith.muli %11, %12 : i32
          %14 = arith.addi %8, %13 : i32
          affine.store %14, %alloc_6[%arg1, %arg2] : memref<1x16xi32>
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %6 = affine.load %alloc_6[%arg1, %arg2] : memref<1x16xi32>
        %7 = arith.extsi %6 : i32 to i64
        %8 = arith.muli %7, %c1561796795_i64 : i64
        %9 = arith.addi %8, %c68719476736_i64 : i64
        %10 = arith.cmpi sge, %6, %c0_i32 : i32
        %11 = arith.select %10, %c1073741824_i64, %c-1073741824_i64 : i64
        %12 = arith.addi %11, %9 : i64
        %13 = arith.shrsi %12, %c37_i64 : i64
        %14 = arith.trunci %13 : i64 to i32
        %15 = arith.addi %14, %c-128_i32 : i32
        %16 = arith.cmpi slt, %15, %c-128_i32 : i32
        %17 = arith.select %16, %c-128_i32, %15 : i32
        %18 = arith.cmpi sgt, %15, %c127_i32 : i32
        %19 = arith.select %18, %c127_i32, %17 : i32
        %20 = arith.trunci %19 : i32 to i8
        affine.store %20, %alloc_7[%arg1, %arg2] : memref<1x16xi8>
      }
    }
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 16 {
        %6 = affine.load %alloc_7[0, %arg2] : memref<1x16xi8>
        %7 = arith.cmpi slt, %6, %c-128_i8 : i8
        %8 = arith.select %7, %c-128_i8, %6 : i8
        %9 = arith.cmpi sgt, %6, %c127_i8 : i8
        %10 = arith.select %9, %c127_i8, %8 : i8
        affine.store %10, %alloc_8[%arg1, %arg2] : memref<1x16xi8>
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<16x1xi8>
    affine.for %arg1 = 0 to 16 {
      affine.for %arg2 = 0 to 1 {
        %6 = affine.load %4[%arg2, %arg1] : memref<1x16xi8>
        affine.store %6, %alloc_9[%arg1, %arg2] : memref<16x1xi8>
      }
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.store %c429_i32, %alloc_10[%arg1, %arg2] : memref<1x1xi32>
      }
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
    %5 = affine.load %alloc_10[0, 0] : memref<1x1xi32>
    affine.store %5, %alloc_11[0, 0] : memref<1x1xi32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 16 {
          %6 = affine.load %alloc_8[%arg1, %arg3] : memref<1x16xi8>
          %7 = affine.load %alloc_9[%arg3, %arg2] : memref<16x1xi8>
          %8 = affine.load %alloc_11[%arg1, %arg2] : memref<1x1xi32>
          %9 = arith.extsi %6 : i8 to i16
          %10 = arith.subi %9, %c-128_i16 : i16
          %11 = arith.extsi %10 : i16 to i32
          %12 = arith.extsi %7 : i8 to i32
          %13 = arith.muli %11, %12 : i32
          %14 = arith.addi %8, %13 : i32
          affine.store %14, %alloc_11[%arg1, %arg2] : memref<1x1xi32>
        }
      }
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi8>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        %6 = affine.load %alloc_11[%arg1, %arg2] : memref<1x1xi32>
        %7 = arith.extsi %6 : i32 to i64
        %8 = arith.muli %7, %c1630361836_i64 : i64
        %9 = arith.addi %8, %c34359738368_i64 : i64
        %10 = arith.cmpi sge, %6, %c0_i32 : i32
        %11 = arith.select %10, %c1073741824_i64, %c-1073741824_i64 : i64
        %12 = arith.addi %11, %9 : i64
        %13 = arith.shrsi %12, %c36_i64 : i64
        %14 = arith.trunci %13 : i64 to i32
        %15 = arith.addi %14, %c5_i32 : i32
        %16 = arith.cmpi slt, %15, %c-128_i32 : i32
        %17 = arith.select %16, %c-128_i32, %15 : i32
        %18 = arith.cmpi sgt, %15, %c127_i32 : i32
        %19 = arith.select %18, %c127_i32, %17 : i32
        %20 = arith.trunci %19 : i32 to i8
        affine.store %20, %alloc_12[%arg1, %arg2] : memref<1x1xi8>
      }
    }
    // CHECK: secret.yield %[[ALLOC:.*]] : memref<1x1xi8>
    // CHECK: return %[[V0]] : !secret.secret<memref<1x1xi8>>
    return %alloc_12 : memref<1x1xi8>
  }
}
