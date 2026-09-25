// RUN: heir-opt --cheddar-bufferize %s | FileCheck %s

// Cheddar DPS ops bufferize like linalg ops: operands become memrefs and the
// result is folded into its destination. Function results become
// `bufferize.result` out-params (buffer-results-to-out-params).

// CHECK: func.func @add
// CHECK: cheddar.add {{.*}} : (!context, memref<!ciphertext>, memref<!ciphertext>, memref<!ciphertext>) -> ()
// CHECK-NEXT: return
func.func @add(%ctx: !cheddar.context, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %result = cheddar.add %ctx, %lhs, %rhs, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// Read-write destinations in a nonstandard operand position.
// CHECK: func.func @mad_unsafe
// CHECK: cheddar.mad_unsafe {{.*}} : (!context, memref<!ciphertext>, memref<!ciphertext>, memref<!constant>) -> ()
// CHECK-NEXT: return
func.func @mad_unsafe(%ctx: !cheddar.context, %acc: tensor<!cheddar.ciphertext>, %input: tensor<!cheddar.ciphertext>, %constant: tensor<!cheddar.constant>) -> tensor<!cheddar.ciphertext> {
  %result = cheddar.mad_unsafe %ctx, %acc, %input, %constant : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.constant>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

// A result equivalent to an argument is dropped; the argument is updated in place.
// CHECK: func.func @prepare_rot_key(%[[UI:.*]]: memref<!user_interface>) {
// CHECK: cheddar.prepare_rot_key %[[UI]] {distance = 7 : i64, maxLevel = 13 : i64} : (memref<!user_interface>) -> ()
// CHECK-NEXT: return
func.func @prepare_rot_key(%ui: tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface> {
  %result = cheddar.prepare_rot_key %ui {distance = 7 : i64, maxLevel = 13 : i64} : (tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
  return %result : tensor<!cheddar.user_interface>
}

// CHECK: func.func @prepare_bootstrap(%[[CTX:.*]]: memref<!boot_context>, %[[UI:.*]]: memref<!user_interface>) {
// CHECK: cheddar.prepare_bootstrap %[[CTX]], %[[UI]] {numSlots = 8 : i64} : (memref<!boot_context>, memref<!user_interface>) -> ()
// CHECK-NEXT: return
func.func @prepare_bootstrap(%ctx: tensor<!cheddar.boot_context>, %ui: tensor<!cheddar.user_interface>) -> (tensor<!cheddar.boot_context>, tensor<!cheddar.user_interface>) {
  %new_ctx, %new_ui = cheddar.prepare_bootstrap %ctx, %ui {numSlots = 8 : i64} : (tensor<!cheddar.boot_context>, tensor<!cheddar.user_interface>) -> (tensor<!cheddar.boot_context>, tensor<!cheddar.user_interface>)
  return %new_ctx, %new_ui : tensor<!cheddar.boot_context>, tensor<!cheddar.user_interface>
}

// CHECK: func.func @decode
// CHECK-SAME: %[[DECODED:[a-zA-Z0-9_]+]]: memref<4xf64>
// CHECK: cheddar.decode %{{.*}}, %{{.*}}, %[[DECODED]] : (!encoder, memref<!plaintext>, memref<4xf64>) -> ()
func.func @decode(%encoder: !cheddar.encoder, %plaintext: tensor<!cheddar.plaintext>, %value: tensor<4xf64>) -> tensor<4xf64> {
  %decoded = cheddar.decode %encoder, %plaintext, %value : (!cheddar.encoder, tensor<!cheddar.plaintext>, tensor<4xf64>) -> tensor<4xf64>
  return %decoded : tensor<4xf64>
}

// A chain of fully overwriting ops reuses one destination, which becomes the
// out-param: no allocation, no copy.
// CHECK: func.func @reuse_sequence
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: cheddar.add %{{.*}}, %{{.*}}, %{{.*}}, %[[OUT]]
// CHECK: cheddar.neg %{{.*}}, %[[OUT]], %[[OUT]]
// CHECK-NOT: memref.copy
func.func @reuse_sequence(%ctx: !cheddar.context, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %empty = tensor.empty() : tensor<!cheddar.ciphertext>
  %sum = cheddar.add %ctx, %lhs, %rhs, %empty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %negated = cheddar.neg %ctx, %sum, %sum : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %negated : tensor<!cheddar.ciphertext>
}

// A distinct tensor.empty destination stays a distinct buffer.
// CHECK: func.func @rescale_fresh
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = memref.alloc(){{.*}} : memref<!ciphertext>
// CHECK: cheddar.add %{{.*}}, %{{.*}}, %{{.*}}, %[[INPUT]]
// CHECK: cheddar.rescale %{{.*}}, %[[INPUT]], %[[OUT]]
func.func @rescale_fresh(%ctx: !cheddar.context, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %inputEmpty = tensor.empty() : tensor<!cheddar.ciphertext>
  %input = cheddar.add %ctx, %lhs, %rhs, %inputEmpty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %outputEmpty = tensor.empty() : tensor<!cheddar.ciphertext>
  %output = cheddar.rescale %ctx, %input, %outputEmpty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %output : tensor<!cheddar.ciphertext>
}

// Passing the same value as input and destination is not a conflict.
// CHECK: func.func @hrot_add_in_place
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: cheddar.hrot_add %{{.*}}, %{{.*}}, %[[OUT]], %[[OUT]], %[[OUT]]
// CHECK-NOT: memref.copy
func.func @hrot_add_in_place(%ctx: !cheddar.context, %evk: !cheddar.evk_map, %lhs: tensor<!cheddar.ciphertext>, %rhs: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %empty = tensor.empty() : tensor<!cheddar.ciphertext>
  %input = cheddar.add %ctx, %lhs, %rhs, %empty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %output = cheddar.hrot_add %ctx, %evk, %input, %input, %input {distance = 2 : i64} : (!cheddar.context, !cheddar.evk_map, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %output : tensor<!cheddar.ciphertext>
}

// A loop carrying a caller-owned input updates it in place; the (equivalent)
// result is dropped.
// CHECK: func.func @loop_in_place(%{{.*}}: !context, %[[INIT:.*]]: memref<!ciphertext>, %{{.*}}: index) {
// CHECK-NOT: memref.copy
// CHECK: scf.for
// CHECK-NEXT: cheddar.neg %{{.*}}, %[[INIT]], %[[INIT]]
func.func @loop_in_place(%ctx: !cheddar.context, %init: tensor<!cheddar.ciphertext>, %upper: index) -> tensor<!cheddar.ciphertext> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %c0 to %upper step %c1 iter_args(%iter = %init) -> tensor<!cheddar.ciphertext> {
    %next = cheddar.neg %ctx, %iter, %iter : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    scf.yield %next : tensor<!cheddar.ciphertext>
  }
  return %result : tensor<!cheddar.ciphertext>
}

// Empty-tensor elimination threads the packed result through the insertion, so
// the producer writes straight into the out-param slot.
// CHECK: func.func @packed_encrypt
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<1x!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: %[[SLOT:[a-zA-Z0-9_]+]] = memref.subview %[[OUT]][0] [1] [1]
// CHECK: cheddar.encrypt %{{.*}}, %{{.*}}, %[[SLOT]]
// CHECK-NOT: memref.copy
func.func @packed_encrypt(%ui: !cheddar.user_interface, %plaintext: tensor<!cheddar.plaintext>) -> tensor<1x!cheddar.ciphertext> {
  %scalarInit = tensor.empty() : tensor<!cheddar.ciphertext>
  %encrypted = cheddar.encrypt %ui, %plaintext, %scalarInit : (!cheddar.user_interface, tensor<!cheddar.plaintext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  %packedInit = tensor.empty() : tensor<1x!cheddar.ciphertext>
  %packed = tensor.insert_slice %encrypted into %packedInit[0] [1] [1] : tensor<!cheddar.ciphertext> into tensor<1x!cheddar.ciphertext>
  return %packed : tensor<1x!cheddar.ciphertext>
}

// CHECK: func.func @loop_packed
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<8x!ciphertext> {bufferize.result}
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK: %[[SLOT:[a-zA-Z0-9_]+]] = memref.subview %[[OUT]][%{{.*}}] [1] [1]
// CHECK: cheddar.add %{{.*}}, %{{.*}}, %{{.*}}, %[[SLOT]]
// CHECK-NOT: memref.copy
func.func @loop_packed(%ctx: !cheddar.context, %input: tensor<!cheddar.ciphertext>) -> tensor<8x!cheddar.ciphertext> {
  %output = tensor.empty() : tensor<8x!cheddar.ciphertext>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%iter = %output) -> tensor<8x!cheddar.ciphertext> {
    %empty = tensor.empty() : tensor<!cheddar.ciphertext>
    %value = cheddar.add %ctx, %input, %input, %empty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    %inserted = tensor.insert_slice %value into %iter[%i] [1] [1] : tensor<!cheddar.ciphertext> into tensor<8x!cheddar.ciphertext>
    scf.yield %inserted : tensor<8x!cheddar.ciphertext>
  }
  return %result : tensor<8x!cheddar.ciphertext>
}

// A returned call result goes through a temporary and one copy into the
// out-param; the EmitC lowering turns the copy of a dead temporary into a move.
func.func private @produce(%ctx: !cheddar.context, %input: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %empty = tensor.empty() : tensor<!cheddar.ciphertext>
  %result = cheddar.neg %ctx, %input, %empty : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}
// CHECK: func.func @forward
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]+]]: memref<!ciphertext> {bufferize.result}
// CHECK: %[[TMP:.*]] = memref.alloc() : memref<!ciphertext>
// CHECK: call @produce({{.*}}, %[[TMP]])
// CHECK: memref.copy %[[TMP]], %[[OUT]]
// CHECK-NEXT: return
func.func @forward(%ctx: !cheddar.context, %input: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
  %result = func.call @produce(%ctx, %input) : (!cheddar.context, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
  return %result : tensor<!cheddar.ciphertext>
}

func.func @setup(%params: !cheddar.parameter) -> tensor<!cheddar.context> {
  %empty = tensor.empty() : tensor<!cheddar.context>
  %ctx = cheddar.create_context %params, %empty : (!cheddar.parameter, tensor<!cheddar.context>) -> tensor<!cheddar.context>
  return %ctx : tensor<!cheddar.context>
}
func.func @keygen(%ctx: tensor<!cheddar.context>) -> (tensor<!cheddar.context>, tensor<!cheddar.user_interface>) {
  %empty = tensor.empty() : tensor<!cheddar.user_interface>
  %ui = cheddar.create_user_interface %ctx, %empty : (tensor<!cheddar.context>, tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
  %ui2 = cheddar.prepare_rot_key %ui {distance = 1 : i64, maxLevel = 1 : i64} : (tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
  return %ctx, %ui2 : tensor<!cheddar.context>, tensor<!cheddar.user_interface>
}
// CHECK: func.func @configure(%[[PARAMS:.*]]: !parameter, %[[CTX:.*]]: memref<!context> {bufferize.result}, %[[UI:.*]]: memref<!user_interface> {bufferize.result})
// CHECK: %[[TMP_CTX:.*]] = memref.alloc() : memref<!context>
// CHECK: call @setup(%[[PARAMS]], %[[TMP_CTX]])
// CHECK: %[[TMP_UI:.*]] = memref.alloc() : memref<!user_interface>
// CHECK: call @keygen(%[[TMP_CTX]], %[[TMP_UI]])
// CHECK: memref.copy %[[TMP_CTX]], %[[CTX]]
// CHECK: memref.copy %[[TMP_UI]], %[[UI]]
// CHECK: return
func.func @configure(%params: !cheddar.parameter) -> (tensor<!cheddar.context>, tensor<!cheddar.user_interface>) {
  %ctx = func.call @setup(%params) : (!cheddar.parameter) -> tensor<!cheddar.context>
  %ctx2, %ui = func.call @keygen(%ctx) : (tensor<!cheddar.context>) -> (tensor<!cheddar.context>, tensor<!cheddar.user_interface>)
  return %ctx2, %ui : tensor<!cheddar.context>, tensor<!cheddar.user_interface>
}
