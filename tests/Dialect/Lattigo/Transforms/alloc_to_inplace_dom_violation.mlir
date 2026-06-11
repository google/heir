// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

!ct = !lattigo.rlwe.ciphertext
!evaluator = !lattigo.bgv.evaluator

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [17179926529], P = [17179967489], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK: func.func @dom_violation
  func.func @dom_violation(%evaluator: !evaluator, %ct_in: !ct, %memref: memref<10x!ct>) -> !ct {
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c10 step %c1 {
      %0 = memref.load %memref[%i] : memref<10x!ct>
      %alloc = lattigo.bgv.add_new %evaluator, %0, %0 : (!evaluator, !ct, !ct) -> !ct
      memref.store %alloc, %memref[%i] : memref<10x!ct>
    }

    // Outside the loop
    // This operation might illegally reuse %alloc if dominance is not checked.
    // We return %ct_in to keep it live, so it cannot be reused here.
    // CHECK: lattigo.bgv.add_new
    %out = lattigo.bgv.add_new %evaluator, %ct_in, %ct_in : (!evaluator, !ct, !ct) -> !ct

    return %ct_in : !ct
  }
}
