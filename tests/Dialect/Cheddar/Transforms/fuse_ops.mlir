// RUN: heir-opt --cheddar-fuse-ops %s | FileCheck %s

!ct = !cheddar.ciphertext

// CHECK: @fuse_hmult_rescale
func.func @fuse_hmult_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK-NOT: cheddar.rescale
  // CHECK: cheddar.hmult
  // CHECK-NOT: rescale = false
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %relin = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  %d2 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.rescale %ctx, %relin, %d2 : (!cheddar.context, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_hmult_no_rescale
func.func @fuse_hmult_no_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize
  // CHECK: cheddar.hmult
  // CHECK-SAME: rescale = false
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_hmult_relinearize_rescale
func.func @fuse_hmult_relinearize_rescale(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> tensor<!ct> {
  // CHECK-NOT: cheddar.mult
  // CHECK-NOT: cheddar.relinearize_rescale
  // CHECK: cheddar.hmult
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.relinearize_rescale %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_rotation_and_conjugation
func.func @fuse_rotation_and_conjugation(
    %ctx: !cheddar.context, %ui: !cheddar.user_interface,
    %input: tensor<!ct>, %other: tensor<!ct>) -> (tensor<!ct>, tensor<!ct>) {
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 3
  // CHECK: cheddar.hconj_add
  %r0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %ui, %input, %r0 <static_distance = 3 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %r1 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated_sum = cheddar.add %ctx, %rotated, %other, %r1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %c0 = bufferization.alloc_tensor() : tensor<!ct>
  %conjugated = cheddar.hconj %ctx, %ui, %input, %c0 : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %c1 = bufferization.alloc_tensor() : tensor<!ct>
  %conjugated_sum = cheddar.add %ctx, %conjugated, %other, %c1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %rotated_sum, %conjugated_sum : tensor<!ct>, tensor<!ct>
}

// CHECK: @fuse_rotation_rhs
func.func @fuse_rotation_rhs(
    %ctx: !cheddar.context, %ui: !cheddar.user_interface,
    %input: tensor<!ct>, %other: tensor<!ct>) -> tensor<!ct> {
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 4
  %r0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %ui, %input, %r0 <static_distance = 4 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %r1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.add %ctx, %other, %rotated, %r1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_conjugation_rhs
func.func @fuse_conjugation_rhs(
    %ctx: !cheddar.context, %ui: !cheddar.user_interface,
    %input: tensor<!ct>, %other: tensor<!ct>) -> tensor<!ct> {
  // CHECK: cheddar.hconj_add
  %c0 = bufferization.alloc_tensor() : tensor<!ct>
  %conjugated = cheddar.hconj %ctx, %ui, %input, %c0 : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %c1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.add %ctx, %other, %conjugated, %c1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// CHECK: @fuse_rotation_dynamic_constant_distance
func.func @fuse_rotation_dynamic_constant_distance(
    %ctx: !cheddar.context, %ui: !cheddar.user_interface,
    %input: tensor<!ct>, %other: tensor<!ct>) -> tensor<!ct> {
  // CHECK: cheddar.hrot_add
  // CHECK-SAME: distance = 5
  %c5 = arith.constant 5 : index
  %r0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %ui, %input, %r0, %c5 : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>, index) -> tensor<!ct>
  %r1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.add %ctx, %rotated, %other, %r1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// Fusing across different contexts would silently change semantics.
// CHECK: @do_not_fuse_different_contexts
func.func @do_not_fuse_different_contexts(
    %ctx0: !cheddar.context, %ctx1: !cheddar.context,
    %ui: !cheddar.user_interface, %input: tensor<!ct>,
    %other: tensor<!ct>) -> tensor<!ct> {
  // CHECK: cheddar.hrot
  // CHECK: cheddar.add
  // CHECK-NOT: cheddar.hrot_add
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx0, %ui, %input, %d0 <static_distance = 2 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.add %ctx1, %rotated, %other, %d1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// Non-constant dynamic distance cannot be fused into hrot_add.
// CHECK: @do_not_fuse_dynamic_non_constant
func.func @do_not_fuse_dynamic_non_constant(
    %ctx: !cheddar.context, %ui: !cheddar.user_interface,
    %input: tensor<!ct>, %other: tensor<!ct>, %dist: index) -> tensor<!ct> {
  // CHECK: cheddar.hrot
  // CHECK: cheddar.add
  // CHECK-NOT: cheddar.hrot_add
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %ui, %input, %d0, %dist : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>, index) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %result = cheddar.add %ctx, %rotated, %other, %d1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %result : tensor<!ct>
}

// If mult has multiple uses, it must not be erased/fused.
// CHECK: @do_not_fuse_mult_multiple_uses
func.func @do_not_fuse_mult_multiple_uses(
    %ctx: !cheddar.context, %lhs: tensor<!ct>, %rhs: tensor<!ct>,
    %key: !cheddar.eval_key) -> (tensor<!ct>, tensor<!ct>) {
  // CHECK: cheddar.mult
  // CHECK: cheddar.relinearize
  // CHECK-NOT: cheddar.hmult
  %d0 = bufferization.alloc_tensor() : tensor<!ct>
  %mult = cheddar.mult %ctx, %lhs, %rhs, %d0 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %d1 = bufferization.alloc_tensor() : tensor<!ct>
  %relin = cheddar.relinearize %ctx, %mult, %key, %d1 : (!cheddar.context, tensor<!ct>, !cheddar.eval_key, tensor<!ct>) -> tensor<!ct>
  return %mult, %relin : tensor<!ct>, tensor<!ct>
}

// If hrot has multiple uses, it must not be erased/fused.
// CHECK: @do_not_fuse_hrot_multiple_uses
func.func @do_not_fuse_hrot_multiple_uses(
    %ctx: !cheddar.context, %ui: !cheddar.user_interface,
    %input: tensor<!ct>, %other: tensor<!ct>) -> (tensor<!ct>, tensor<!ct>) {
  // CHECK: cheddar.hrot
  // CHECK: cheddar.add
  // CHECK-NOT: cheddar.hrot_add
  %r0 = bufferization.alloc_tensor() : tensor<!ct>
  %rotated = cheddar.hrot %ctx, %ui, %input, %r0 <static_distance = 3 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %r1 = bufferization.alloc_tensor() : tensor<!ct>
  %sum = cheddar.add %ctx, %rotated, %other, %r1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %rotated, %sum : tensor<!ct>, tensor<!ct>
}

// If hconj has multiple uses, it must not be erased/fused.
// CHECK: @do_not_fuse_hconj_multiple_uses
func.func @do_not_fuse_hconj_multiple_uses(
    %ctx: !cheddar.context, %ui: !cheddar.user_interface,
    %input: tensor<!ct>, %other: tensor<!ct>) -> (tensor<!ct>, tensor<!ct>) {
  // CHECK: cheddar.hconj
  // CHECK: cheddar.add
  // CHECK-NOT: cheddar.hconj_add
  %c0 = bufferization.alloc_tensor() : tensor<!ct>
  %conjugated = cheddar.hconj %ctx, %ui, %input, %c0 : (!cheddar.context, !cheddar.user_interface, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  %c1 = bufferization.alloc_tensor() : tensor<!ct>
  %sum = cheddar.add %ctx, %conjugated, %other, %c1 : (!cheddar.context, tensor<!ct>, tensor<!ct>, tensor<!ct>) -> tensor<!ct>
  return %conjugated, %sum : tensor<!ct>, tensor<!ct>
}
