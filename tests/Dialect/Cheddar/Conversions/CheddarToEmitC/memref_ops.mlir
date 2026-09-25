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
