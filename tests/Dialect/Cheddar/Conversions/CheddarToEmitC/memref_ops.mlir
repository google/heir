// RUN: heir-opt --convert-to-emitc=filter-dialects=cheddar --cheddar-emitc-boundary --reconcile-unrealized-casts %s | FileCheck %s

// memref ops that lower to structured EmitC loops and assignments.

// Deallocating a payload resets it to a fresh object with a move-assignment
// from an inlined `T()` literal.
// CHECK: func.func @dealloc_scalar
// CHECK: %[[V:.*]] = "emitc.variable"{{.*}} -> !emitc.lvalue<!emitc.opaque<"Ciphertext<word>">>
// CHECK: emitc.member_call_opaque %arg0 "Neg"(%[[V]], %arg1)
// CHECK: %[[FRESH:.*]] = emitc.literal "Ciphertext<word>()" : !emitc.opaque<"Ciphertext<word>">
// CHECK: emitc.assign %[[FRESH]] : !emitc.opaque<"Ciphertext<word>"> to %[[V]]
func.func @dealloc_scalar(%ctx: !cheddar.context, %in: memref<!cheddar.ciphertext>,
                          %out: memref<!cheddar.ciphertext> {bufferize.result}) {
  %tmp = memref.alloc() : memref<!cheddar.ciphertext>
  cheddar.neg %ctx, %in, %tmp : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  cheddar.neg %ctx, %tmp, %out : (!cheddar.context, memref<!cheddar.ciphertext>, memref<!cheddar.ciphertext>) -> ()
  memref.dealloc %tmp : memref<!cheddar.ciphertext>
  return
}

// A rank-1 payload array is reset element by element.
// CHECK: func.func @dealloc_array
// CHECK: %[[A:.*]] = "emitc.variable"{{.*}} -> !emitc.array<3x!emitc.opaque<"Ciphertext<word>">>
// CHECK: %[[UB:.*]] = emitc.literal "3" : !emitc.size_t
// CHECK: emitc.for %[[I:.*]] = %{{.*}} to %[[UB]] step %{{.*}} : !emitc.size_t {
// CHECK:   %[[SLOT:.*]] = subscript %[[A]][%[[I]]]
// CHECK:   %[[FRESH:.*]] = literal "Ciphertext<word>()"
// CHECK:   assign %[[FRESH]] : !emitc.opaque<"Ciphertext<word>"> to %[[SLOT]]
func.func @dealloc_array() {
  %tmp = memref.alloc() : memref<3x!cheddar.ciphertext>
  memref.dealloc %tmp : memref<3x!cheddar.ciphertext>
  return
}

// A cleartext copy between flat pointers is an element-wise loop.
// CHECK: func.func @copy_primitive(%[[SRC:.*]]: !emitc.ptr<f32>, %[[DST:.*]]: !emitc.ptr<f32>)
// CHECK: %[[UB:.*]] = emitc.literal "8" : !emitc.size_t
// CHECK: emitc.for %[[I:.*]] = %{{.*}} to %[[UB]] step %{{.*}} : !emitc.size_t {
// CHECK:   %[[FROM:.*]] = subscript %[[SRC]][%[[I]]]
// CHECK:   %[[VAL:.*]] = load %[[FROM]]
// CHECK:   %[[TO:.*]] = subscript %[[DST]][%[[I]]]
// CHECK:   assign %[[VAL]] : f32 to %[[TO]]
func.func @copy_primitive(%src: memref<2x4xf32>, %dst: memref<2x4xf32>) {
  memref.copy %src, %dst : memref<2x4xf32> to memref<2x4xf32>
  return
}

// A layout cast of a payload buffer is a no-op: users index the original array.
// CHECK: func.func @payload_cast(%[[CTX:.*]]: !emitc.ptr<!emitc.opaque<"Context<word>">>, %[[M:.*]]: !emitc.array<2x!emitc.opaque<"const Ciphertext<word>">>, %[[OUT:.*]]: !emitc.opaque<"Ciphertext<word>&">
// CHECK-NOT: memref.cast
// CHECK: %[[SLOT:.*]] = emitc.subscript %[[M]][%{{.*}}]
// CHECK: emitc.member_call_opaque %[[CTX]] "Neg"(%[[OUT]], %[[SLOT]])
func.func @payload_cast(%ctx: !cheddar.context, %m: memref<2x!cheddar.ciphertext>,
                        %out: memref<!cheddar.ciphertext> {bufferize.result}) {
  %cast = memref.cast %m : memref<2x!cheddar.ciphertext> to memref<2x!cheddar.ciphertext, strided<[1]>>
  %slot = memref.subview %cast[1] [1] [1] : memref<2x!cheddar.ciphertext, strided<[1]>> to memref<!cheddar.ciphertext, strided<[], offset: 1>>
  cheddar.neg %ctx, %slot, %out : (!cheddar.context, memref<!cheddar.ciphertext, strided<[], offset: 1>>, memref<!cheddar.ciphertext>) -> ()
  return
}

// A layout cast of an evaluation-key buffer is a no-op, as for the other
// payload types.
// CHECK: func.func @cast_eval_key(%[[CTX:.*]]: !emitc.ptr<!emitc.opaque<"Context<word>">>, %[[CT:.*]]: !emitc.opaque<"const Ciphertext<word>&">, %[[KEY:.*]]: !emitc.opaque<"const EvaluationKey<word>&">, %[[OUT:.*]]: !emitc.opaque<"Ciphertext<word>&">
// CHECK-NOT: memref.cast
// CHECK: emitc.member_call_opaque %[[CTX]] "Relinearize"(%[[OUT]], %[[CT]], %[[KEY]])
func.func @cast_eval_key(%ctx: !cheddar.context, %ct: memref<!cheddar.ciphertext>,
                         %key: memref<!cheddar.eval_key, strided<[]>>,
                         %out: memref<!cheddar.ciphertext> {bufferize.result}) {
  %c = memref.cast %key : memref<!cheddar.eval_key, strided<[]>> to memref<!cheddar.eval_key>
  %k = memref.load %c[] : memref<!cheddar.eval_key>
  cheddar.relinearize %ctx, %ct, %k, %out : (!cheddar.context, memref<!cheddar.ciphertext>, !cheddar.eval_key, memref<!cheddar.ciphertext>) -> ()
  return
}

// Loading from an array of evaluation keys subscripts with the array's
// element type.
// CHECK: func.func @load_eval_key_array(%[[CTX:.*]]: !emitc.ptr<!emitc.opaque<"Context<word>">>, %[[CT:.*]]: !emitc.opaque<"const Ciphertext<word>&">, %[[KEYS:.*]]: !emitc.array<2x!emitc.opaque<"const EvaluationKey<word>">>, %[[I:.*]]: !emitc.size_t, %[[OUT:.*]]: !emitc.opaque<"Ciphertext<word>&">
// CHECK: %[[KEY:.*]] = emitc.subscript %[[KEYS]][%[[I]]] : (!emitc.array<2x!emitc.opaque<"const EvaluationKey<word>">>, !emitc.size_t) -> !emitc.lvalue<!emitc.opaque<"const EvaluationKey<word>">>
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.member_call_opaque %[[CTX]] "Relinearize"(%[[OUT]], %[[CT]], %[[KEY]])
func.func @load_eval_key_array(%ctx: !cheddar.context, %ct: memref<!cheddar.ciphertext>,
                               %keys: memref<2x!cheddar.eval_key>, %i: index,
                               %out: memref<!cheddar.ciphertext> {bufferize.result}) {
  %k = memref.load %keys[%i] : memref<2x!cheddar.eval_key>
  cheddar.relinearize %ctx, %ct, %k, %out : (!cheddar.context, memref<!cheddar.ciphertext>, !cheddar.eval_key, memref<!cheddar.ciphertext>) -> ()
  return
}

// A loaded evaluation key passed to a function call is the element lvalue
// itself, not a cast of it.
// CHECK: func.func @call_eval_key_array(%[[KEYS:.*]]: !emitc.array<2x!emitc.opaque<"const EvaluationKey<word>">>, %[[I:.*]]: !emitc.size_t)
// CHECK: %[[KEY:.*]] = emitc.subscript %[[KEYS]][%[[I]]]
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.call_opaque "use_eval_key"(%[[KEY]])
func.func private @use_eval_key(!cheddar.eval_key)
func.func @call_eval_key_array(%keys: memref<2x!cheddar.eval_key>, %i: index) {
  %k = memref.load %keys[%i] : memref<2x!cheddar.eval_key>
  func.call @use_eval_key(%k) : (!cheddar.eval_key) -> ()
  return
}
