// RUN: heir-opt --secret-distribute-generic --secret-to-cggi -cse %s | FileCheck %s

// This test was produced by running
//   heir-opt --yosys-optimizer --canonicalize tests/yosys_optimizer/add_one.mlir

module {
  func.func @add_one(%arg0: !secret.secret<i8>) -> !secret.secret<i8> {
    // CHECK-NOT: comb
    // CHECK-NOT: secret.generic
    // CHECK-NOT: secret.cast
    // CHECK-COUNT-15: cggi.lut3
    %true = arith.constant true
    %false = arith.constant false
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = secret.cast %arg0 : !secret.secret<i8> to !secret.secret<memref<8xi1>>
    %alloc = memref.alloc() : memref<8xi1>
    memref.store %true, %alloc[%c0] : memref<8xi1>
    memref.store %false, %alloc[%c1] : memref<8xi1>
    memref.store %false, %alloc[%c2] : memref<8xi1>
    memref.store %false, %alloc[%c3] : memref<8xi1>
    memref.store %false, %alloc[%c4] : memref<8xi1>
    memref.store %false, %alloc[%c5] : memref<8xi1>
    memref.store %false, %alloc[%c6] : memref<8xi1>
    memref.store %false, %alloc[%c7] : memref<8xi1>
    %1 = secret.generic ins(%0 : !secret.secret<memref<8xi1>>) {
    ^bb0(%arg1: memref<8xi1>):
      %3 = memref.load %arg1[%c0] : memref<8xi1>
      %4 = memref.load %alloc[%c0] : memref<8xi1>
      %5 = comb.truth_table %3, %4, %false -> 8 : ui8
      %6 = memref.load %arg1[%c1] : memref<8xi1>
      %7 = memref.load %alloc[%c1] : memref<8xi1>
      %8 = comb.truth_table %5, %6, %7 -> 150 : ui8
      %9 = comb.truth_table %5, %6, %7 -> 23 : ui8
      %10 = memref.load %arg1[%c2] : memref<8xi1>
      %11 = memref.load %alloc[%c2] : memref<8xi1>
      %12 = comb.truth_table %9, %10, %11 -> 43 : ui8
      %13 = memref.load %arg1[%c3] : memref<8xi1>
      %14 = memref.load %alloc[%c3] : memref<8xi1>
      %15 = comb.truth_table %12, %13, %14 -> 43 : ui8
      %16 = memref.load %arg1[%c4] : memref<8xi1>
      %17 = memref.load %alloc[%c4] : memref<8xi1>
      %18 = comb.truth_table %15, %16, %17 -> 43 : ui8
      %19 = memref.load %arg1[%c5] : memref<8xi1>
      %20 = memref.load %alloc[%c5] : memref<8xi1>
      %21 = comb.truth_table %18, %19, %20 -> 43 : ui8
      %22 = memref.load %arg1[%c6] : memref<8xi1>
      %23 = memref.load %alloc[%c6] : memref<8xi1>
      %24 = comb.truth_table %21, %22, %23 -> 105 : ui8
      %25 = comb.truth_table %21, %22, %23 -> 43 : ui8
      %26 = memref.load %arg1[%c7] : memref<8xi1>
      %27 = memref.load %alloc[%c7] : memref<8xi1>
      %28 = comb.truth_table %25, %26, %27 -> 105 : ui8
      %29 = comb.truth_table %3, %4, %false -> 6 : ui8
      %30 = comb.truth_table %9, %10, %11 -> 105 : ui8
      %31 = comb.truth_table %12, %13, %14 -> 105 : ui8
      %32 = comb.truth_table %15, %16, %17 -> 105 : ui8
      %33 = comb.truth_table %18, %19, %20 -> 105 : ui8
      %alloc_0 = memref.alloc() : memref<8xi1>
      memref.store %28, %alloc_0[%c0] : memref<8xi1>
      memref.store %24, %alloc_0[%c1] : memref<8xi1>
      memref.store %33, %alloc_0[%c2] : memref<8xi1>
      memref.store %32, %alloc_0[%c3] : memref<8xi1>
      memref.store %31, %alloc_0[%c4] : memref<8xi1>
      memref.store %30, %alloc_0[%c5] : memref<8xi1>
      memref.store %8, %alloc_0[%c6] : memref<8xi1>
      memref.store %29, %alloc_0[%c7] : memref<8xi1>
      secret.yield %alloc_0 : memref<8xi1>
    } -> !secret.secret<memref<8xi1>>
    %2 = secret.cast %1 : !secret.secret<memref<8xi1>> to !secret.secret<i8>
    return %2 : !secret.secret<i8>
  }
}
