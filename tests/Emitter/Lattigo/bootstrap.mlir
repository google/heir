// RUN: heir-translate %s --emit-lattigo | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: heir-translate %s --emit-lattigo --lattigo-bootstrap-declared-scale-log2=8 | FileCheck %s --check-prefixes=CHECK,WIDE

// CHECK: func Test_bootstrap(
// CHECK-SAME: [[BOOT_EVAL:[^, ]+]] *bootstrapping.Evaluator, [[EVAL:[^, ]+]] *ckks.Evaluator, [[CT_IN:[^, ]+]] *rlwe.Ciphertext, [[CT_OTHER:[^, ]+]] *rlwe.Ciphertext
// The default (s = 0) form: a bare Bootstrap call, no declared-scale bundle.
// DEFAULT: [[CT_OUT:[^, ]+]], err{{.*}} := [[BOOT_EVAL]].Bootstrap([[CT_IN]].CopyNew())
// DEFAULT: if err{{.*}} != nil {
// DEFAULT:   panic(err{{.*}})
// DEFAULT: }
// DEFAULT-NOT: .Evaluator.Mul
// DEFAULT-NOT: ResidualParameters

// The declared-scale bundle (s = 8): declare the input scale as Delta*2^8
// (metadata only, on a copy), bootstrap, then restore the true message with
// an exact integer scalar multiply by 2^8 and re-stamp the default scale.
// WIDE: [[CT_WIDE:[^, ]+]] := [[CT_IN]].CopyNew()
// WIDE-NEXT: [[CT_WIDE]].Scale = [[CT_WIDE]].Scale.Mul(rlwe.NewScale(uint64(256)))
// WIDE-NEXT: [[CT_OUT:[^, ]+]], err{{.*}} := [[BOOT_EVAL]].Bootstrap([[CT_WIDE]])
// WIDE: if err{{.*}} != nil {
// WIDE:   panic(err{{.*}})
// WIDE: }
// WIDE: err{{.*}} := [[BOOT_EVAL]].Evaluator.Mul([[CT_OUT]], big.NewInt(256), [[CT_OUT]])
// WIDE: if err{{.*}} != nil {
// WIDE:   panic(err{{.*}})
// WIDE: }
// WIDE: [[CT_OUT]].Scale = [[BOOT_EVAL]].ResidualParameters.DefaultScale()
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
