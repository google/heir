// RUN: heir-opt --partial-unroll-for-level-consumption=force-max-level=4 %s | FileCheck %s

// CHECK: @affine_loop

// The pre-existing loop preamble
// CHECK: arith.addi
// CHECK-NEXT: mgmt.level_reduce_min

// Unrolled loop
// CHECK-NEXT: affine.for {{.*}} = 1 to 9 step 4
// iteration 1
// CHECK-NEXT:  mgmt.bootstrap
// CHECK-NEXT:  arith.muli
// CHECK-NEXT:  mgmt.relinearize
// CHECK-NEXT:  mgmt.modreduce

// iteration 2
// CHECK-NEXT:  arith.muli
// CHECK-NEXT:  mgmt.relinearize
// CHECK-NEXT:  mgmt.modreduce

// iteration 3
// CHECK-NEXT:  arith.muli
// CHECK-NEXT:  mgmt.relinearize
// CHECK-NEXT:  mgmt.modreduce

// iteration 4
// CHECK-NEXT:  arith.muli
// CHECK-NEXT:  mgmt.relinearize
// CHECK-NEXT:  mgmt.modreduce

// CHECK-NEXT:  mgmt.level_reduce_min
// CHECK-NEXT:  affine.yield
// CHECK-NEXT:  }

// cleanup loop
// CHECK-NEXT:   affine.for [[arg1:.*]] = 9 to 12
// CHECK-NEXT:    mgmt.bootstrap
// CHECK-NEXT:    arith.muli
// CHECK-NEXT:    mgmt.relinearize
// CHECK-NEXT:    mgmt.modreduce
// CHECK-NEXT:    mgmt.level_reduce_min
// CHECK-NEXT:    affine.yield
func.func @affine_loop(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c1_i32 = arith.constant 1 : i32
  %0 = secret.generic(%arg0: !secret.secret<i32>) {
  ^body(%input0: i32):
    %1 = arith.addi %c1_i32, %input0 : i32
    %2 = mgmt.level_reduce_min %1 : i32
    %3 = affine.for %arg1 = 1 to 12 iter_args(%arg2 = %2) -> (i32) {
      %4 = mgmt.bootstrap %arg2 : i32
      %5 = arith.muli %4, %input0 : i32
      %6 = mgmt.relinearize %5 : i32
      %7 = mgmt.modreduce %6 : i32
      %8 = mgmt.level_reduce_min %7 : i32
      affine.yield %8 : i32
    }
    secret.yield %3 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
