// This tests the functionality of --extract-loop-body.

// RUN: heir-opt --extract-loop-body="min-body-size=2" %s > %t
// RUN: FileCheck %s < %t

// CHECK: module
module {
  // CHECK: memref.global
  memref.global "private" constant @__constant_513xi16 : memref<513xi16> = dense<"0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000100010001000100010001000200020002000200020003000300030004000400050005000600060007000800080009000A000B000C000E000F00110012001400160018001A001D002000230026002A002E00330038003D00430049005000580061006A0074007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F007F00">
  // CHECK: func.func @loop_body() -> memref<25x20x8xi8>
  func.func @loop_body() -> memref<25x20x8xi8> {
    // CHECK: [[MEMGLOBAL:%[a-z0-9_]+]] = memref.get_global
    %1 = memref.get_global @__constant_513xi16 : memref<513xi16>
    %c-128_i8 = arith.constant -128 : i8
    %c127_i8 = arith.constant 127 : i8
    %c0 = arith.constant 0 : i8
    // CHECK: [[MEMREF1:%[a-z0-9_]+]] = memref.alloc
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<25x20x8xi8>
    // This loop body should not be extracted, since it does not follow a store
    // and load pattern.
    // CHECK: affine.for
    affine.for %arg1 = 0 to 25 {
      // CHECK: affine.for
      affine.for %arg2 = 0 to 20 {
        // CHECK: affine.for
        affine.for %arg3 = 0 to 8 {
          // CHECK-NOT: func.call
          // CHECK: affine.store
          affine.store %c127_i8, %alloc_6[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
        }
      }
    }
    // CHECK: [[MEMREF2:%[a-z0-9_]+]] = memref.alloc
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<25x20x8xi8>
    // CHECK: affine.for
    affine.for %arg1 = 0 to 25 {
      // CHECK: affine.for
      affine.for %arg2 = 0 to 20 {
        // CHECK: affine.for
        affine.for %arg3 = 0 to 8 {
          // CHECK: affine.load [[MEMREF1]]
          %98 = affine.load %alloc_6[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
          // CHECK-NEXT: [[V1:%[a-z0-9_]+]] = func.call @for_
          %99 = arith.cmpi slt, %98, %c-128_i8 : i8
          %100 = arith.select %99, %c-128_i8, %98 : i8
          %101 = arith.cmpi sgt, %98, %c127_i8 : i8
          %102 = arith.select %101, %c127_i8, %100 : i8
          // CHECK-NEXT: affine.store [[V1]], [[MEMREF2]]
          affine.store %102, %alloc_7[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
        }
      }
    }
    // CHECK: [[MEMREF3:%[a-z0-9_]+]] = memref.alloc
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<25x20x8xi8>
    // This loop tests that the statements used to compute the memref.load stay
    // in the for loop body.
    // CHECK: affine.for
    affine.for %arg1 = 0 to 25 {
      // CHECK: affine.for
      affine.for %arg2 = 0 to 20 {
        // CHECK: affine.for
        affine.for %arg3 = 0 to 8 {
          // CHECK-NEXT: arith.index_cast
          %2 = arith.index_cast %arg1 : index to i16
          // CHECK-NEXT: arith.index_cast
          %3 = arith.index_cast %arg2 : index to i16
          // CHECK-NEXT: arith.muli
          %4 = arith.muli %2, %3 : i16
          // CHECK-NEXT: arith.index_cast
          %5 = arith.index_cast %4 : i16 to index
          // CHECK-NEXT: memref.load
          %97 = memref.load %1[%5] : memref<513xi16>
          // CHECK-NEXT: affine.load
          %98 = affine.load %alloc_7[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
          // CHECK-NEXT: [[V2:%[a-z0-9_]+]] = func.call @for_
          %99 = arith.trunci %97 : i16 to i8
          %101 = arith.cmpi sgt, %98, %c127_i8 : i8
          %102 = arith.select %101, %99, %98 : i8
          // CHECK-NEXT: affine.store [[V2]], [[MEMREF3]]
          affine.store %102, %alloc_8[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
        }
      }
    }
    // CHECK: [[MEMREF3:%[a-z0-9_]+]] = memref.alloc
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<25x20x8xi8>
    // CHECK: affine.for
    affine.for %arg1 = 0 to 25 {
      // CHECK: affine.for
      affine.for %arg2 = 0 to 20 {
        // CHECK: affine.for
        affine.for %arg3 = 0 to 8 {
          // This tests the minimum loop body size - should not extract.
          // CHECK-NOT: func.call
          %98 = affine.load %alloc_6[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
          %99 = affine.load %alloc_7[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
          %100 = arith.addi %98, %99 : i8
          affine.store %100, %alloc_9[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
        }
      }
    }
    // This tests the minimum loop size - should not extract.
    // CHECK: affine.for
    affine.for %arg1 = 0 to 1 {
      // CHECK: affine.for
      affine.for %arg2 = 0 to 1 {
        // CHECK: affine.for
        affine.for %arg3 = 0 to 2 {
          // CHECK-NOT: func.call
          %98 = affine.load %alloc_6[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
          %99 = arith.cmpi slt, %98, %c-128_i8 : i8
          %100 = arith.select %99, %c-128_i8, %98 : i8
          %101 = arith.cmpi sgt, %98, %c127_i8 : i8
          %102 = arith.select %101, %c127_i8, %100 : i8
          affine.store %100, %alloc_9[%arg1, %arg2, %arg3] : memref<25x20x8xi8>
        }
      }
    }
    // CHECK: return
    return %alloc_9 : memref<25x20x8xi8>
  }
}
