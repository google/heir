// RUN: heir-opt --poulpy-bufferize %s | FileCheck %s

// Poulpy DPS ops bufferize like linalg ops: operands become memrefs and each
// result is folded into its destination. Function results stay results.

!module = !poulpy.module
!scratch = !poulpy.scratch
!ct = tensor<!poulpy.ciphertext>
!pt = tensor<!poulpy.plaintext>
!sk = !poulpy.secret_key
!tsk = !poulpy.tensor_key
!akm = !poulpy.automorphism_key_map

// A fresh destination becomes an allocation that is returned.
// CHECK: func.func @fresh_dst
// CHECK-SAME: -> memref<!poulpy.ciphertext>
// CHECK: %[[DST:.*]] = memref.alloc()
// CHECK-NEXT: poulpy.add %{{.*}}, %[[DST]], %{{.*}}, %{{.*}}, %{{.*}} : (!poulpy.module, memref<!poulpy.ciphertext>, memref<!poulpy.ciphertext>, memref<!poulpy.ciphertext>, !poulpy.scratch) -> ()
// CHECK-NEXT: return %[[DST]]
func.func @fresh_dst(%m: !module, %s: !scratch, %a: !ct, %b: !ct) -> !ct {
  %e = tensor.empty() : !ct
  %r = poulpy.add %m, %e, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> !ct
  return %r : !ct
}

// An _assign op updates its destination's buffer in place; an _into op never
// writes into the buffer of one of its inputs.
// CHECK: func.func @chain
// CHECK: %[[SUM:.*]] = memref.alloc()
// CHECK-NEXT: poulpy.add %{{.*}}, %[[SUM]],
// CHECK-NEXT: poulpy.mul_assign %{{.*}}, %[[SUM]],
// CHECK-NEXT: %[[ROT:.*]] = memref.alloc()
// CHECK-NEXT: poulpy.rotate %{{.*}}, %[[ROT]], %[[SUM]],
// CHECK-NEXT: return %[[ROT]]
func.func @chain(%m: !module, %s: !scratch, %tsk: !tsk, %akm: !akm, %a: !ct, %b: !ct) -> !ct {
  %e0 = tensor.empty() : !ct
  %sum = poulpy.add %m, %e0, %a, %b, %s : (!module, !ct, !ct, !ct, !scratch) -> !ct
  %prod = poulpy.mul_assign %m, %sum, %b, %tsk, %s : (!module, !ct, !ct, !tsk, !scratch) -> !ct
  %e1 = tensor.empty() : !ct
  %rot = poulpy.rotate %m, %e1, %prod, %akm, %s <k = 1> : (!module, !ct, !ct, !akm, !scratch) -> !ct
  return %rot : !ct
}

// An _assign op on an argument that is still live updates a copy.
// CHECK: func.func @assign_on_live_arg(%{{.*}}: !poulpy.module, %{{.*}}: !poulpy.scratch, %[[A:.*]]: memref<!poulpy.ciphertext>, %{{.*}}: memref<!poulpy.ciphertext>) -> memref<!poulpy.ciphertext>
// CHECK: %[[COPY:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[A]], %[[COPY]]
// CHECK-NEXT: poulpy.add_assign %{{.*}}, %[[COPY]],
// CHECK-NEXT: return %[[COPY]]
func.func @assign_on_live_arg(%m: !module, %s: !scratch, %a: !ct, %b: !ct) -> (!ct, !ct) {
  %r = poulpy.add_assign %m, %a, %b, %s : (!module, !ct, !ct, !scratch) -> !ct
  return %r, %a : !ct, !ct
}

// A result equivalent to an argument is dropped; the argument is updated in place.
// CHECK: func.func @update_arg(%{{.*}}: !poulpy.module, %{{.*}}: !poulpy.scratch, %[[A:.*]]: memref<!poulpy.ciphertext>, %{{.*}}: memref<!poulpy.ciphertext>) {
// CHECK-NEXT: poulpy.add_assign %{{.*}}, %[[A]],
// CHECK-NEXT: return
func.func @update_arg(%m: !module, %s: !scratch, %a: !ct, %b: !ct) -> !ct {
  %r = poulpy.add_assign %m, %a, %b, %s : (!module, !ct, !ct, !scratch) -> !ct
  return %r : !ct
}

// decode writes both of its slot vectors.
// CHECK: func.func @roundtrip
// CHECK-SAME: -> (memref<4xf64>, memref<4xf64>)
// CHECK: %[[PT:.*]] = memref.alloc() {{.*}} : memref<!poulpy.plaintext>
// CHECK-NEXT: poulpy.encode %{{.*}}, %[[PT]],
// CHECK-NEXT: %[[CT:.*]] = memref.alloc() {{.*}} : memref<!poulpy.ciphertext>
// CHECK-NEXT: poulpy.encrypt %{{.*}}, %[[CT]], %[[PT]],
// CHECK-NEXT: %[[PT2:.*]] = memref.alloc() {{.*}} : memref<!poulpy.plaintext>
// CHECK-NEXT: poulpy.decrypt %{{.*}}, %[[PT2]], %[[CT]],
// CHECK-NEXT: %[[RE:.*]] = memref.alloc() {{.*}} : memref<4xf64>
// CHECK-NEXT: %[[IM:.*]] = memref.alloc() {{.*}} : memref<4xf64>
// CHECK-NEXT: poulpy.decode %{{.*}}, %[[RE]], %[[IM]], %[[PT2]],
// CHECK-NEXT: return %[[RE]], %[[IM]]
func.func @roundtrip(%m: !module, %s: !scratch, %sk: !sk, %re: tensor<4xf64>, %im: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>) {
  %pt_empty = tensor.empty() : !pt
  %pt = poulpy.encode %m, %pt_empty, %re, %im, %s <logDelta = 45, logBudget = 20, base2k = 52> : (!module, !pt, tensor<4xf64>, tensor<4xf64>, !scratch) -> !pt
  %ct_empty = tensor.empty() : !ct
  %ct = poulpy.encrypt %m, %ct_empty, %pt, %sk, %s <base2k = 52, ctk = 300> : (!module, !ct, !pt, !sk, !scratch) -> !ct
  %pt2_empty = tensor.empty() : !pt
  %pt2 = poulpy.decrypt %m, %pt2_empty, %ct, %sk, %s : (!module, !pt, !ct, !sk, !scratch) -> !pt
  %re_empty = tensor.empty() : tensor<4xf64>
  %im_empty = tensor.empty() : tensor<4xf64>
  %re_out, %im_out = poulpy.decode %m, %re_empty, %im_empty, %pt2, %s : (!module, tensor<4xf64>, tensor<4xf64>, !pt, !scratch) -> (tensor<4xf64>, tensor<4xf64>)
  return %re_out, %im_out : tensor<4xf64>, tensor<4xf64>
}
