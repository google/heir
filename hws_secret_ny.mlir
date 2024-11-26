module attributes {tf_saved_model.semantics} {
  memref.global "private" constant @__constant_1x3xi8 : memref<1x3xi8> = dense<[[-39, 59, 39]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi32_0 : memref<3xi32> = dense<[-729, 1954, 610]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3x3xi8 : memref<3x3xi8> = dense<[[-12, 26, -19], [9, 25, 33], [-12, 36, -32]]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3xi32 : memref<3xi32> = dense<[0, 0, -5438]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3x1xi8 : memref<3x1xi8> = dense<[[-9], [-54], [57]]> {alignment = 64 : i64}
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
    %0 = memref.get_global @__constant_3x1xi8 : memref<3x1xi8>
    %1 = memref.get_global @__constant_3xi32 : memref<3xi32>
    %2 = memref.get_global @__constant_3x3xi8 : memref<3x3xi8>
    %3 = memref.get_global @__constant_3xi32_0 : memref<3xi32>
    %4 = memref.get_global @__constant_1x3xi8 : memref<1x3xi8>
    %5 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3xi32>
      secret.yield %alloc : memref<1x3xi32>
    } -> !secret.secret<memref<1x3xi32>>
    affine.for %arg1 = 0 to 3 {
      secret.generic ins(%5 : !secret.secret<memref<1x3xi32>>) {
      ^bb0(%arg2: memref<1x3xi32>):
        %8 = memref.load %1[%arg1] : memref<3xi32>
        memref.store %8, %arg2[%c0, %arg1] : memref<1x3xi32>
        secret.yield
      }
    }
    %6:2 = secret.generic {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x3xi32>
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x1xi32>
      memref.store %c429_i32, %alloc_0[%c0, %c0] : memref<1x1xi32>
      secret.yield %alloc, %alloc_0 : memref<1x3xi32>, memref<1x1xi32>
    } -> (!secret.secret<memref<1x3xi32>>, !secret.secret<memref<1x1xi32>>)
    affine.for %arg1 = 0 to 3 {
      %8 = secret.generic ins(%6#0 : !secret.secret<memref<1x3xi32>>) {
      ^bb0(%arg2: memref<1x3xi32>):
        %9 = memref.load %3[%arg1] : memref<3xi32>
        memref.store %9, %arg2[%c0, %arg1] : memref<1x3xi32>
        secret.yield %9 : i32
      } -> !secret.secret<i32>
      affine.for %arg2 = 0 to 3 {
        secret.generic ins(%arg0, %5, %6#0 : !secret.secret<memref<1x1xi8, strided<[?, ?], offset: ?>>>, !secret.secret<memref<1x3xi32>>, !secret.secret<memref<1x3xi32>>) {
        ^bb0(%arg3: memref<1x1xi8, strided<[?, ?], offset: ?>>, %arg4: memref<1x3xi32>, %arg5: memref<1x3xi32>):
          %9 = memref.load %0[%arg2, %c0] : memref<3x1xi8>
          %10 = memref.load %arg3[%c0, %c0] : memref<1x1xi8, strided<[?, ?], offset: ?>>
          %11 = memref.load %arg4[%c0, %arg2] : memref<1x3xi32>
          %12 = arith.extsi %10 : i8 to i16
          %13 = arith.subi %12, %c-128_i16 : i16
          %14 = arith.extui %13 : i16 to i32
          %15 = arith.extsi %9 : i8 to i32
          %16 = arith.muli %14, %15 : i32
          %17 = arith.addi %11, %16 : i32
          memref.store %17, %arg4[%c0, %arg2] : memref<1x3xi32>
          %18 = arith.extsi %17 : i32 to i64
          %19 = arith.muli %18, %c2039655736_i64 : i64
          %20 = arith.addi %19, %c137438953472_i64 : i64
          %21 = arith.cmpi sge, %17, %c0_i32 : i32
          %22 = arith.select %21, %c1073741824_i64, %c-1073741824_i64 : i64
          %23 = arith.addi %22, %20 : i64
          %24 = arith.shrsi %23, %c38_i64 : i64
          %25 = arith.trunci %24 : i64 to i32
          %26 = arith.addi %25, %c-128_i32 : i32
          %27 = arith.maxsi %26, %c-128_i32 : i32
          %28 = arith.minsi %27, %c127_i32 : i32
          %29 = arith.trunci %28 : i32 to i8
          %30 = memref.load %2[%arg1, %arg2] : memref<3x3xi8>
          %31 = memref.load %arg5[%c0, %arg1] : memref<1x3xi32>
          %32 = arith.extsi %29 : i8 to i16
          %33 = arith.subi %32, %c-128_i16 : i16
          %34 = arith.extui %33 : i16 to i32
          %35 = arith.extsi %30 : i8 to i32
          %36 = arith.muli %34, %35 : i32
          %37 = arith.addi %31, %36 : i32
          memref.store %37, %arg5[%c0, %arg1] : memref<1x3xi32>
          secret.yield
        }
      }
      secret.generic ins(%6#0, %6#1 : !secret.secret<memref<1x3xi32>>, !secret.secret<memref<1x1xi32>>) {
      ^bb0(%arg2: memref<1x3xi32>, %arg3: memref<1x1xi32>):
        %9 = memref.load %arg2[%c0, %arg1] : memref<1x3xi32>
        %10 = arith.extsi %9 : i32 to i64
        %11 = arith.muli %10, %c1561796795_i64 : i64
        %12 = arith.addi %11, %c68719476736_i64 : i64
        %13 = arith.cmpi sge, %9, %c0_i32 : i32
        %14 = arith.select %13, %c1073741824_i64, %c-1073741824_i64 : i64
        %15 = arith.addi %14, %12 : i64
        %16 = arith.shrsi %15, %c37_i64 : i64
        %17 = arith.trunci %16 : i64 to i32
        %18 = arith.addi %17, %c-128_i32 : i32
        %19 = arith.maxsi %18, %c-128_i32 : i32
        %20 = arith.minsi %19, %c127_i32 : i32
        %21 = arith.trunci %20 : i32 to i8
        %22 = memref.load %4[%c0, %arg1] : memref<1x3xi8>
        %23 = memref.load %arg3[%c0, %c0] : memref<1x1xi32>
        %24 = arith.extsi %21 : i8 to i16
        %25 = arith.subi %24, %c-128_i16 : i16
        %26 = arith.extui %25 : i16 to i32
        %27 = arith.extsi %22 : i8 to i32
        %28 = arith.muli %26, %27 : i32
        %29 = arith.addi %23, %28 : i32
        memref.store %29, %arg3[%c0, %c0] : memref<1x1xi32>
        secret.yield
      }
    }
    %7 = secret.generic ins(%5, %6#0, %6#1 : !secret.secret<memref<1x3xi32>>, !secret.secret<memref<1x3xi32>>, !secret.secret<memref<1x1xi32>>) {
    ^bb0(%arg1: memref<1x3xi32>, %arg2: memref<1x3xi32>, %arg3: memref<1x1xi32>):
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1xi8>
      %8 = memref.load %arg3[%c0, %c0] : memref<1x1xi32>
      %9 = arith.extsi %8 : i32 to i64
      %10 = arith.muli %9, %c1630361836_i64 : i64
      %11 = arith.addi %10, %c34359738368_i64 : i64
      %12 = arith.cmpi sge, %8, %c0_i32 : i32
      %13 = arith.select %12, %c1073741824_i64, %c-1073741824_i64 : i64
      %14 = arith.addi %13, %11 : i64
      %15 = arith.shrsi %14, %c36_i64 : i64
      %16 = arith.trunci %15 : i64 to i32
      %17 = arith.addi %16, %c5_i32 : i32
      %18 = arith.maxsi %17, %c-128_i32 : i32
      %19 = arith.minsi %18, %c127_i32 : i32
      %20 = arith.trunci %19 : i32 to i8
      memref.store %20, %alloc[%c0, %c0] : memref<1x1xi8>
      memref.dealloc %arg1 : memref<1x3xi32>
      memref.dealloc %arg2 : memref<1x3xi32>
      memref.dealloc %arg3 : memref<1x1xi32>
      secret.yield %alloc : memref<1x1xi8>
    } -> !secret.secret<memref<1x1xi8>>
    return %7 : !secret.secret<memref<1x1xi8>>
  }
}
