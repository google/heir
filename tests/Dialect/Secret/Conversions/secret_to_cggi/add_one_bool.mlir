// RUN: heir-opt --secret-distribute-generic --secret-to-cggi -cse %s | FileCheck %s

// This test was produced by running
//   heir-opt --yosys-optimizer="mode=Boolean" --canonicalize tests/yosys_optimizer/add_one.mlir

module {
  func.func @add_one(%arg0: !secret.secret<i8>) -> !secret.secret<i8> {
    // CHECK-NOT: comb
    // CHECK-NOT: secret.generic
    // CHECK-NOT: secret.cast
    // CHECK-COUNT-20: cggi
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = secret.cast %arg0 : !secret.secret<i8> to !secret.secret<memref<8xi1>>
    %1 = secret.generic ins(%0 : !secret.secret<memref<8xi1>>) {
    ^bb0(%arg1: memref<8xi1>):
      %3 = memref.load %arg1[%c0] : memref<8xi1>
      %4 = comb.inv %3 : i1
      %5 = memref.load %arg1[%c2] : memref<8xi1>
      %6 = comb.inv %5 : i1
      %7 = memref.load %arg1[%c1] : memref<8xi1>
      %8 = comb.and %3, %7 : i1
      %9 = comb.xnor %6, %8 : i1
      %10 = comb.and %5, %8 : i1
      %11 = memref.load %arg1[%c3] : memref<8xi1>
      %12 = comb.and %11, %10 : i1
      %13 = comb.inv %11 : i1
      %14 = comb.xnor %13, %10 : i1
      %15 = memref.load %arg1[%c4] : memref<8xi1>
      %16 = comb.and %15, %12 : i1
      %17 = comb.inv %15 : i1
      %18 = comb.xnor %17, %12 : i1
      %19 = memref.load %arg1[%c5] : memref<8xi1>
      %20 = comb.and %19, %16 : i1
      %21 = comb.inv %19 : i1
      %22 = comb.xnor %21, %16 : i1
      %23 = memref.load %arg1[%c6] : memref<8xi1>
      %24 = comb.and %23, %20 : i1
      %25 = comb.inv %23 : i1
      %26 = comb.xnor %25, %20 : i1
      %27 = memref.load %arg1[%c7] : memref<8xi1>
      %28 = comb.inv %27 : i1
      %29 = comb.xnor %28, %24 : i1
      %30 = comb.xor %3, %7 : i1
      %alloc = memref.alloc() : memref<8xi1>
      memref.store %4, %alloc[%c0] : memref<8xi1>
      memref.store %30, %alloc[%c1] : memref<8xi1>
      memref.store %9, %alloc[%c2] : memref<8xi1>
      memref.store %14, %alloc[%c3] : memref<8xi1>
      memref.store %18, %alloc[%c4] : memref<8xi1>
      memref.store %22, %alloc[%c5] : memref<8xi1>
      memref.store %26, %alloc[%c6] : memref<8xi1>
      memref.store %29, %alloc[%c7] : memref<8xi1>
      secret.yield %alloc : memref<8xi1>
    } -> !secret.secret<memref<8xi1>>
    %2 = secret.cast %1 : !secret.secret<memref<8xi1>> to !secret.secret<i8>
    return %2 : !secret.secret<i8>
  }
}
