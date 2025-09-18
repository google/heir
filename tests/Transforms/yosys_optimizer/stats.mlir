// RUN: heir-opt --secret-distribute-generic=distribute-through="affine.for" --yosys-optimizer="unroll-factor=4 print-stats=true" -o /dev/null %s 2>&1 | FileCheck %s

!in_ty = !secret.secret<tensor<10xi8>>
!out_ty = !secret.secret<tensor<10xi8>>

// Computes the set of partial cumulative sums of the input array
func.func @cumulative_sums(%arg0: !in_ty) -> (!out_ty) {
  %0 = secret.generic() {
  ^bb0:
    %tensor = tensor.empty() : tensor<10xi8>
    secret.yield %tensor : tensor<10xi8>
  } -> !out_ty

  %after_first_elem = secret.generic(%arg0: !in_ty, %0 : !out_ty) {
  ^bb0(%input: tensor<10xi8>, %alloc: tensor<10xi8>):
    %c0 = arith.constant 0 : index
    %val = tensor.extract %input[%c0] : tensor<10xi8>
    %out = tensor.insert %val into %alloc[%c0] : tensor<10xi8>
    secret.yield %out : tensor<10xi8>
  } -> !out_ty

  %out = affine.for %i = 1 to 10 iter_args(%iter = %after_first_elem) -> (!out_ty) {
    %secret_out = secret.generic(%arg0: !in_ty, %iter : !out_ty) {
    ^bb0(%input: tensor<10xi8>, %accum: tensor<10xi8>):
      %c1 = arith.constant 1 : index
      %i_minus_one = arith.subi %i, %c1 : index
      %next_val = tensor.extract %input[%i] : tensor<10xi8>
      %prev_sum = tensor.extract %accum[%i_minus_one] : tensor<10xi8>
      %next_sum  = arith.addi %prev_sum, %next_val : i8
      %out = tensor.insert %next_sum into %accum[%i] : tensor<10xi8>
      secret.yield %out : tensor<10xi8>
    } -> !out_ty
    affine.yield %secret_out : !out_ty
  }

  return %out : !out_ty
}

// CHECK: Starting arith op count: 4
// CHECK-NEXT: Ending cell count: 143
// CHECK-NEXT: Ratio: 3.575000e+01
