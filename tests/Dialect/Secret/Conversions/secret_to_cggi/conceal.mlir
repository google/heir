// RUN: heir-opt --secret-distribute-generic --secret-to-cggi %s | FileCheck %s

// CHECK-LABEL: @conceal
// CHECK-SAME: ([[ARG:%.*]]: i3) -> [[TY:memref<3x!lwe.lwe_ciphertext<.*>>]]
// CHECK-NEXT: [[C1:%.*]] = arith.constant 1 : i3
// CHECK-NEXT: [[ALLOC:%.*]] = memref.alloc() : [[TY]]
// CHECK-NEXT: affine.for [[ITER:%.*]] = 0 to 3 {
// CHECK-NEXT:   [[IDX:%.*]] = arith.index_cast [[ITER]] : index to i3
// CHECK-NEXT:   [[SHL:%.*]] = arith.shli [[C1]], [[IDX]] : i3
// CHECK-NEXT:   [[AND:%.*]] = arith.andi [[ARG]], [[SHL]] : i3
// CHECK-NEXT:   [[SHR:%.*]] = arith.shrsi [[AND]], [[IDX]] : i3
// CHECK-NEXT:   [[TRUNC:%.*]] = arith.trunci [[SHR]] : i3 to i1
// CHECK-NEXT:   [[ENCODE:%.*]] = lwe.encode [[TRUNC]]
// CHECK-NEXT:   [[ENCRYPT:%.*]] = lwe.trivial_encrypt [[ENCODE]]
// CHECK-NEXT:   memref.store [[ENCRYPT]], [[ALLOC]][[[ITER]]] : [[TY]]
// CHECK-NEXT: }
// CHECK-NEXT: return [[ALLOC]] : [[TY]]
func.func @conceal(%arg0: i3) -> !secret.secret<i3> {
  %0 = secret.conceal %arg0 : i3 -> !secret.secret<i3>
  func.return %0 : !secret.secret<i3>
}
