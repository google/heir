// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

// Regression test: in-place storage reuse must not cross CKKS levels.
//
// `CallerProvidedStorageInfo::getAvailableStorage` (lib/Utils/AllocToInPlaceUtils.h)
// accepts any dead buffer whose consumed-level depth is <= the op result's
// depth, rejecting only `storage depth > op depth`. A buffer last written at a
// *shallower* depth can therefore become the destination of a *deeper* result.
// The IR stays type-correct, but the emitted Lattigo destination carries the
// wrong runtime level: the value is later multiplied at (or below) the modulus
// floor and lattigo panics with
//   "cannot Rescale: input Ciphertext level is too low".
// This shows up on deep composite-sign ReLUs, where many same-typed buffers of
// differing depth are live at once.
//
// Reuse is only sound at an exactly matching depth.

!evaluator = !lattigo.ckks.evaluator
!ct = !lattigo.rlwe.ciphertext

// %rot sits at depth 0 and is dead once %deep is produced, so it is the
// candidate storage for %deep — whose depth is 4. Reusing it emits
// `drop_level %rot, %rot` and pins the depth-4 value into a depth-0 buffer;
// the depth-4 result must keep its own allocation instead.
//
// CHECK: func.func @no_reuse_across_levels
func.func @no_reuse_across_levels(%evaluator: !evaluator, %ct: !ct) -> !ct {
  // CHECK: %[[rot:.*]] = lattigo.ckks.rotate_new
  %rot = lattigo.ckks.rotate_new %evaluator, %ct {static_shift = 4} : (!evaluator, !ct) -> !ct
  // CHECK: %[[shallow:.*]] = lattigo.rlwe.drop_level_new %{{.*}} {levelToDrop = 2 : i64}
  %shallow = lattigo.rlwe.drop_level_new %evaluator, %ct { levelToDrop = 2 } : (!evaluator, !ct) -> !ct
  // The depth-4 result must NOT be written into the depth-0 %rot buffer, i.e.
  // this stays `drop_level_new` rather than becoming `drop_level %rot, %rot`.
  // CHECK: %[[deep:.*]] = lattigo.rlwe.drop_level_new %{{.*}}, %[[rot]] {levelToDrop = 4 : i64}
  %deep = lattigo.rlwe.drop_level_new %evaluator, %rot { levelToDrop = 4 } : (!evaluator, !ct) -> !ct
  // Same-depth reuse is still expected: the add reuses %deep, not %rot.
  // CHECK: lattigo.ckks.add %{{.*}}, %[[shallow]], %[[deep]], %[[deep]]
  %sum = lattigo.ckks.add_new %evaluator, %shallow, %deep : (!evaluator, !ct, !ct) -> !ct
  return %sum : !ct
}

// The same mismatch followed by the multiply/rescale that actually detonates:
// with %deep pinned into the depth-0 %rot buffer, the mul runs against a
// destination whose runtime level no longer matches the level analysis, and the
// rescale below is the op that panics.
//
// CHECK: func.func @no_reuse_across_levels_before_rescale
func.func @no_reuse_across_levels_before_rescale(%evaluator: !evaluator, %ct: !ct) -> !ct {
  // CHECK: %[[rot2:.*]] = lattigo.ckks.rotate_new
  %rot = lattigo.ckks.rotate_new %evaluator, %ct {static_shift = 4} : (!evaluator, !ct) -> !ct
  // CHECK: %[[deep2:.*]] = lattigo.rlwe.drop_level_new %{{.*}}, %[[rot2]] {levelToDrop = 4 : i64}
  %deep = lattigo.rlwe.drop_level_new %evaluator, %rot { levelToDrop = 4 } : (!evaluator, !ct) -> !ct
  // CHECK: %[[prod:.*]] = lattigo.ckks.mul %{{.*}}, %[[deep2]], %[[deep2]], %[[deep2]]
  %prod = lattigo.ckks.mul_new %evaluator, %deep, %deep : (!evaluator, !ct, !ct) -> !ct
  // CHECK: lattigo.ckks.rescale_new %{{.*}}, %[[prod]]
  %out = lattigo.ckks.rescale_new %evaluator, %prod : (!evaluator, !ct) -> !ct
  return %out : !ct
}
