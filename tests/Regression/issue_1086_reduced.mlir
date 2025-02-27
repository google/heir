// RUN: heir-opt --secret-to-cggi %s | FileCheck %s

// CHECK-LABEL: @trivial_loop
// CHECK-NOT: secret
func.func @trivial_loop(%arg0: !secret.secret<memref<2xi3>>, %arg1: !secret.secret<i3>) -> !secret.secret<i3> {
  %c0 = arith.constant 0 : index
  %0 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %arg1) -> (!secret.secret<i3>) {
    %1 = secret.generic ins(%arg0 : !secret.secret<memref<2xi3>>) {
    ^bb0(%arg4: memref<2xi3>):
      %4 = memref.load %arg4[%c0] : memref<2xi3>
      secret.yield %4 : i3
    } -> !secret.secret<i3>
    %2 = secret.generic {
      %alloc = memref.alloc() : memref<3xi1>
      secret.yield %alloc : memref<3xi1>
    } -> !secret.secret<memref<3xi1>>
    %3 = secret.cast %2 : !secret.secret<memref<3xi1>> to !secret.secret<i3>
    affine.yield %3 : !secret.secret<i3>
  }
  return %0 : !secret.secret<i3>
}

// CHECK-LABEL: @sum
// CHECK-NOT: secret
func.func @sum(%arg0: !secret.secret<memref<2xi3>>) -> !secret.secret<i3> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_i3 = arith.constant 0 : i3
  %0 = secret.conceal %c0_i3 : i3 -> !secret.secret<i3>
  %1 = affine.for %arg1 = 0 to 2 iter_args(%arg2 = %0) -> (!secret.secret<i3>) {
    %2 = secret.cast %arg0 : !secret.secret<memref<2xi3>> to !secret.secret<memref<6xi1>>
    %3 = secret.generic ins(%2 : !secret.secret<memref<6xi1>>) {
    ^bb0(%arg3: memref<6xi1>):
      %8 = memref.load %arg3[%c1] : memref<6xi1>
      secret.yield %8 : i1
    } -> !secret.secret<i1>
    %4 = secret.generic ins(%2 : !secret.secret<memref<6xi1>>) {
    ^bb0(%arg3: memref<6xi1>):
      %8 = memref.load %arg3[%c2] : memref<6xi1>
      secret.yield %8 : i1
    } -> !secret.secret<i1>
    %5 = secret.generic ins(%3, %4 : !secret.secret<i1>, !secret.secret<i1>) {
    ^bb0(%arg3: i1, %arg4: i1):
      %8 = comb.truth_table %true, %arg3, %arg4 -> 1 : ui8
      secret.yield %8 : i1
    } -> !secret.secret<i1>
    %6 = secret.generic {
      %alloc = memref.alloc() : memref<3xi1>
      secret.yield %alloc : memref<3xi1>
    } -> !secret.secret<memref<3xi1>>
    secret.generic ins(%5, %6 : !secret.secret<i1>, !secret.secret<memref<3xi1>>) {
    ^bb0(%arg3: i1, %arg4: memref<3xi1>):
      memref.store %arg3, %arg4[%c0] : memref<3xi1>
      secret.yield
    }
    secret.generic ins(%5, %6 : !secret.secret<i1>, !secret.secret<memref<3xi1>>) {
    ^bb0(%arg3: i1, %arg4: memref<3xi1>):
      memref.store %arg3, %arg4[%c1] : memref<3xi1>
      secret.yield
    }
    secret.generic ins(%5, %6 : !secret.secret<i1>, !secret.secret<memref<3xi1>>) {
    ^bb0(%arg3: i1, %arg4: memref<3xi1>):
      memref.store %arg3, %arg4[%c2] : memref<3xi1>
      secret.yield
    }
    %7 = secret.cast %6 : !secret.secret<memref<3xi1>> to !secret.secret<i3>
    affine.yield %7 : !secret.secret<i3>
  }
  return %1 : !secret.secret<i3>
}
