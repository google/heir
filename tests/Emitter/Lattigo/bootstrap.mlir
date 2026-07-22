// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// CHECK: func Test_bootstrap(
// CHECK-SAME: [[BOOT_EVAL:[^, ]+]] *bootstrapping.Evaluator, [[EVAL:[^, ]+]] *ckks.Evaluator, [[CT_IN:[^, ]+]] *rlwe.Ciphertext, [[CT_OTHER:[^, ]+]] *rlwe.Ciphertext
// CHECK:   [[CT_OUT:[^, ]+]], err{{.*}} := [[BOOT_EVAL]].Bootstrap([[CT_IN]])
// CHECK:   if err{{.*}} != nil {
// CHECK:     panic(err{{.*}})
// CHECK:   }
// Use '=' instead of ':=' for DropLevel's CopyNew
// CHECK:   [[CT_OUT]] = [[CT_OTHER]].CopyNew()
// CHECK:   [[EVAL]].DropLevel([[CT_OUT]], 2)
// CHECK:   return [[CT_OUT]]
module attributes {scheme.ckks} {
  func.func @test_bootstrap(%boot_eval: !lattigo.ckks.bootstrapping_evaluator, %eval: !lattigo.ckks.evaluator, %ct: !lattigo.rlwe.ciphertext, %ct_other: !lattigo.rlwe.ciphertext) -> (!lattigo.rlwe.ciphertext) {
    %bootstrapped = lattigo.ckks.bootstrap %boot_eval, %ct : (!lattigo.ckks.bootstrapping_evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
    %reduced = lattigo.rlwe.drop_level %eval, %ct_other, %bootstrapped {levelToDrop = 2 : i64} : (!lattigo.ckks.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
    return %reduced : !lattigo.rlwe.ciphertext
  }
}
