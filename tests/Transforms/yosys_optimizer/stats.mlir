// RUN: heir-opt --secret-distribute-generic=distribute-through="affine.for" --yosys-optimizer="unroll-factor=4 print-stats=true" -o /dev/null %s 2>&1 | FileCheck %s

!in_ty = !secret.secret<memref<10xi8>>
!out_ty = !secret.secret<memref<10xi8>>

// Computes the set of partial cumulative sums of the input array
func.func @cumulative_sums(%arg0: !in_ty) -> (!out_ty) {
  %0 = secret.generic() {
  ^bb0:
    %memref = memref.alloc() : memref<10xi8>
    secret.yield %memref : memref<10xi8>
  } -> !out_ty

  secret.generic(%arg0: !in_ty, %0 : !out_ty) {
  ^bb0(%input: memref<10xi8>, %alloc: memref<10xi8>):
    %c0 = arith.constant 0 : index
    %val = memref.load %input[%c0] : memref<10xi8>
    memref.store %val, %alloc[%c0] : memref<10xi8>
    secret.yield
  }

  affine.for %i = 1 to 10 {
    secret.generic(%arg0: !in_ty, %0 : !out_ty) {
    ^bb0(%input: memref<10xi8>, %accum: memref<10xi8>):
      %c1 = arith.constant 1 : index
      %i_minus_one = arith.subi %i, %c1 : index
      %next_val = memref.load %input[%i] : memref<10xi8>
      %prev_sum = memref.load %accum[%i_minus_one] : memref<10xi8>
      %next_sum  = arith.addi %prev_sum, %next_val : i8
      memref.store %next_sum, %accum[%i] : memref<10xi8>
      secret.yield
    }
  }

  return %0 : !out_ty
}

// CHECK: Starting arith op count: 4
// CHECK-NEXT: Ending cell count: 60
// CHECK-NEXT: Ratio: 1.500000e+01
