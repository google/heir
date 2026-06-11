// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

!ct = !lattigo.rlwe.ciphertext
!evaluator = !lattigo.bgv.evaluator

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [17179926529], P = [17179967489], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK: func.func @loop_rotate
  func.func @loop_rotate(%evaluator: !evaluator, %memref: memref<10x!ct>) {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    // CHECK: scf.for
    scf.for %i = %c0 to %c10 step %c1 {
      %0 = memref.load %memref[%i] : memref<10x!ct>
      // CHECK: lattigo.bgv.rotate_columns %{{.*}}, %{{.*}}, %{{.*}}
      // CHECK-NOT: lattigo.bgv.rotate_columns_new
      %1 = lattigo.bgv.rotate_columns_new %evaluator, %0 {static_shift = 1} : (!evaluator, !ct) -> !ct
      memref.store %1, %memref[%i] : memref<10x!ct>
    }
    return
  }
}
