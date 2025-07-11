// RUN: heir-opt --secret-distribute-generic --secret-to-cggi --split-input-file %s | FileCheck %s

// CHECK: @conceal
// CHECK-SAME: ([[ARG:%.*]]: i3) -> [[TY:memref<3x!.*>]]
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

// -----

// CHECK: @conceal_memref
// CHECK-SAME: ([[ARG:%.*]]: memref<3xi1>) -> [[TY:memref<3x!.*>]]
func.func @conceal_memref(%arg0: memref<3xi1>) -> !secret.secret<memref<3xi1>> {
  // CHECK-COUNT-3: memref.store
  // CHECK-NOT: memref.store
  // CHECK: return
  %0 = secret.conceal %arg0 : memref<3xi1> -> !secret.secret<memref<3xi1>>
  func.return %0 : !secret.secret<memref<3xi1>>
}

// -----

// CHECK: @conceal_memref_iN
// CHECK-SAME: ([[ARG:%.*]]: memref<3xi2>) -> [[TY:memref<3x2x!.*>]]
func.func @conceal_memref_iN(%arg0: memref<3xi2>) -> !secret.secret<memref<3xi2>> {
  // CHECK: memref.load
  // CHECK: affine.for [[ITER:%.*]] = 0 to 2
  // CHECK: memref.store
  // CHECK-NOT: memref.store
  // CHECK: }
  // CHECK: memref.load
  // CHECK: affine.for [[ITER:%.*]] = 0 to 2
  // CHECK: memref.store
  // CHECK-NOT: memref.store
  // CHECK: }
  // CHECK: memref.load
  // CHECK: affine.for [[ITER:%.*]] = 0 to 2
  // CHECK: memref.store
  // CHECK-NOT: memref.store
  // CHECK: }
  %0 = secret.conceal %arg0 : memref<3xi2> -> !secret.secret<memref<3xi2>>
  func.return %0 : !secret.secret<memref<3xi2>>
}

// -----

// CHECK: @conceal_i1
// CHECK-SAME: ([[ARG:%.*]]: i1) -> [[TY:!.*]]
func.func @conceal_i1(%arg0: i1) -> !secret.secret<i1> {
  // CHECK: lwe.encode
  // CHECK-NEXT: lwe.trivial_encrypt
  // CHECK-NEXT: return
  %0 = secret.conceal %arg0 : i1 -> !secret.secret<i1>
  func.return %0 : !secret.secret<i1>
}

// -----

// CHECK: @conceal_MxN
// CHECK-SAME: ([[ARG:%.*]]: memref<3x3xi1>) -> [[TY:memref<3x3x!.*>]]
func.func @conceal_MxN(%arg0: memref<3x3xi1>) -> !secret.secret<memref<3x3xi1>> {
  // CHECK-COUNT-9: memref.load
  // CHECK: return
  %0 = secret.conceal %arg0 : memref<3x3xi1> -> !secret.secret<memref<3x3xi1>>
  func.return %0 : !secret.secret<memref<3x3xi1>>
}
