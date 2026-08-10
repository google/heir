// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// CHECK: func Test_call_reuse(
// CHECK-SAME: [[EVAL:[^, ]+]] *ckks.Evaluator, [[CT_OTHER:[^, ]+]] *rlwe.Ciphertext
// CHECK:   [[CT:[^, ]+]] := Produce()
// Use '=' instead of ':=' for DropLevel's CopyNew
// CHECK:   [[CT]] = [[CT_OTHER]].CopyNew()
// CHECK:   [[EVAL]].DropLevel([[CT]], 2)
// CHECK:   return [[CT]]

module attributes {scheme.ckks} {
  func.func private @produce() -> !lattigo.rlwe.ciphertext

  func.func @test_call_reuse(%eval: !lattigo.ckks.evaluator, %ct_other: !lattigo.rlwe.ciphertext) -> (!lattigo.rlwe.ciphertext) {
    %called = func.call @produce() : () -> !lattigo.rlwe.ciphertext
    %reduced = lattigo.rlwe.drop_level %eval, %ct_other, %called {levelToDrop = 2 : i64} : (!lattigo.ckks.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
    return %reduced : !lattigo.rlwe.ciphertext
  }
}
