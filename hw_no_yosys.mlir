module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_1x16xi8 : memref<1x16xi8> = dense<[[-39, 59, 39, 21, 28, -32, -34, -35, 15, 27, -59, -41, 18, -35, -7, 127]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16xi32_0 : memref<16xi32> = dense<[-729, 1954, 610, 0, 241, -471, -35, -867, 571, 581, 4260, 3943, 591, 0, -889, -5103]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x16xi8 : memref<16x16xi8> = dense<"0xF41AED091921F424E021EFBCF7F5FA1903DCD20206F9F402FFFAEFF1EFD327E1FB27DDEBDBE4051A17FC241215EF1EE410FE14DA1CF8F3F1EFE2F309E3E9EDE3E415070B041B1AFEEB01DE21E60BEC03230A22241E2703E60324FFC011F8FCF1110CF5E0F30717E5E8EDFADCE823FB07DDFBFD0014261117E7F111EA0226040425211D0ADB1DDC2001FAE3370BF11A16EF1CE703E01602032118092ED9E5140BEA1AFCD81300C4D8ECD9FE0D1920D8D6E21FE9D7CAE2DDC613E7043E000114C7DBE71515F506D61ADC0922FE080213EF191EE209FDF314DDDA20D90FE3F9F7EEE924E629000716E21E0D23D3DDF714FA0822262109080F0BE012F47FDC58E526"> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[0, 0, -5438, -5515, -1352, -1500, -4152, -84, 3396, 0, 1981, -5581, 0, -6964, 3407, -7217]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16x1xi8 : memref<16x1xi8> = dense<[[-9], [-54], [57], [71], [104], [115], [98], [99], [64], [-26], [127], [25], [-82], [68], [95], [86]]> {alignment = 64 : i64}
  func.func @main(%arg0: !secret.secret<memref<1x1xi8, strided<[?, ?], offset: ?>>> {iree.identifier = "serving_default_dense_input:0", tf_saved_model.index_path = ["dense_input"]}) -> (!secret.secret<memref<1x1xi8>> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %c-128_i16 = arith.constant -128 : i16
    %c0 = arith.constant 0 : index
    %c429_i32 = arith.constant 429 : i32
    %c34359738368_i64 = arith.constant 34359738368 : i64
    %c36_i64 = arith.constant 36 : i64
    %c1630361836_i64 = arith.constant 1630361836 : i64
    %c68719476736_i64 = arith.constant 68719476736 : i64
    %c37_i64 = arith.constant 37 : i64
    %c1561796795_i64 = arith.constant 1561796795 : i64
    %c137438953472_i64 = arith.constant 137438953472 : i64
    %c38_i64 = arith.constant 38 : i64
    %c2039655736_i64 = arith.constant 2039655736 : i64
    %c5_i32 = arith.constant 5 : i32
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
    %5 = secret.generic ins(%arg0 : !secret.secret<memref<1x1xi8, strided<[?, ?], offset: ?>>>) {
    ^bb0(%arg1: memref<1x1xi8, strided<[?, ?], offset: ?>>):
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      affine.for %arg2 = 0 to 16 {
        %19 = memref.load %1[%arg2] : memref<16xi32>
        memref.store %19, %alloc[%c0, %arg2] : memref<1x16xi32>
      }
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x16xi32>
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
      memref.store %c429_i32, %alloc_1[%c0, %c0] : memref<1x1xi32>
      affine.for %arg2 = 0 to 16 {
        %19 = memref.load %3[%arg2] : memref<16xi32>
        memref.store %19, %alloc_0[%c0, %arg2] : memref<1x16xi32>
        affine.for %arg3 = 0 to 16 {
          %41 = memref.load %0[%arg3, %c0] : memref<16x1xi8>
          %42 = memref.load %arg1[%c0, %c0] : memref<1x1xi8, strided<[?, ?], offset: ?>>
          %43 = memref.load %alloc[%c0, %arg3] : memref<1x16xi32>
          %44 = arith.extsi %42 : i8 to i16
          %45 = arith.subi %44, %c-128_i16 : i16
          %46 = arith.extui %45 : i16 to i32
          %47 = arith.extsi %41 : i8 to i32
          %48 = arith.muli %46, %47 : i32
          %49 = arith.addi %43, %48 : i32
          memref.store %49, %alloc[%c0, %arg3] : memref<1x16xi32>
          %50 = arith.extsi %49 : i32 to i64
          %51 = arith.muli %50, %c2039655736_i64 : i64
          %52 = arith.addi %51, %c137438953472_i64 : i64
          %53 = arith.cmpi sge, %49, %c0_i32 : i32
          %54 = arith.select %53, %c1073741824_i64, %c-1073741824_i64 : i64
          %55 = arith.addi %54, %52 : i64
          %56 = arith.shrsi %55, %c38_i64 : i64
          %57 = arith.trunci %56 : i64 to i32
          %58 = arith.addi %57, %c-128_i32 : i32
          %59 = arith.maxsi %58, %c-128_i32 : i32
          %60 = arith.minsi %59, %c127_i32 : i32
          %61 = arith.trunci %60 : i32 to i8
          %62 = memref.load %2[%arg2, %arg3] : memref<16x16xi8>
          %63 = memref.load %alloc_0[%c0, %arg2] : memref<1x16xi32>
          %64 = arith.extsi %61 : i8 to i16
          %65 = arith.subi %64, %c-128_i16 : i16
          %66 = arith.extui %65 : i16 to i32
          %67 = arith.extsi %62 : i8 to i32
          %68 = arith.muli %66, %67 : i32
          %69 = arith.addi %63, %68 : i32
          memref.store %69, %alloc_0[%c0, %arg2] : memref<1x16xi32>
        }
        %20 = memref.load %alloc_0[%c0, %arg2] : memref<1x16xi32>
        %21 = arith.extsi %20 : i32 to i64
        %22 = arith.muli %21, %c1561796795_i64 : i64
        %23 = arith.addi %22, %c68719476736_i64 : i64
        %24 = arith.cmpi sge, %20, %c0_i32 : i32
        %25 = arith.select %24, %c1073741824_i64, %c-1073741824_i64 : i64
        %26 = arith.addi %25, %23 : i64
        %27 = arith.shrsi %26, %c37_i64 : i64
        %28 = arith.trunci %27 : i64 to i32
        %29 = arith.addi %28, %c-128_i32 : i32
        %30 = arith.maxsi %29, %c-128_i32 : i32
        %31 = arith.minsi %30, %c127_i32 : i32
        %32 = arith.trunci %31 : i32 to i8
        %33 = memref.load %4[%c0, %arg2] : memref<1x16xi8>
        %34 = memref.load %alloc_1[%c0, %c0] : memref<1x1xi32>
        %35 = arith.extsi %32 : i8 to i16
        %36 = arith.subi %35, %c-128_i16 : i16
        %37 = arith.extui %36 : i16 to i32
        %38 = arith.extsi %33 : i8 to i32
        %39 = arith.muli %37, %38 : i32
        %40 = arith.addi %34, %39 : i32
        memref.store %40, %alloc_1[%c0, %c0] : memref<1x1xi32>
      }
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi8>
      %6 = memref.load %alloc_1[%c0, %c0] : memref<1x1xi32>
      %7 = arith.extsi %6 : i32 to i64
      %8 = arith.muli %7, %c1630361836_i64 : i64
      %9 = arith.addi %8, %c34359738368_i64 : i64
      %10 = arith.cmpi sge, %6, %c0_i32 : i32
      %11 = arith.select %10, %c1073741824_i64, %c-1073741824_i64 : i64
      %12 = arith.addi %11, %9 : i64
      %13 = arith.shrsi %12, %c36_i64 : i64
      %14 = arith.trunci %13 : i64 to i32
      %15 = arith.addi %14, %c5_i32 : i32
      %16 = arith.maxsi %15, %c-128_i32 : i32
      %17 = arith.minsi %16, %c127_i32 : i32
      %18 = arith.trunci %17 : i32 to i8
      memref.store %18, %alloc_2[%c0, %c0] : memref<1x1xi8>
      memref.dealloc %alloc : memref<1x16xi32>
      memref.dealloc %alloc_0 : memref<1x16xi32>
      memref.dealloc %alloc_1 : memref<1x1xi32>
      secret.yield %alloc_2 : memref<1x1xi8>
    } -> !secret.secret<memref<1x1xi8>>
    return %5 : !secret.secret<memref<1x1xi8>>
  }
}
