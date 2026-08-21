// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

// A chebyshev evaluation consumes bit_width(degree) levels, so its result sits
// far below its operand. Reusing it as the storage for a higher-level op would
// silently truncate that op's result to the chebyshev level, because lattigo
// evaluates at min(operand, output) level.

!ckks_evaluator = !lattigo.ckks.evaluator
!params = !lattigo.ckks.parameter
!poly_eval = !lattigo.ckks.polynomial_evaluator
!pt = !lattigo.rlwe.plaintext
!ct = !lattigo.rlwe.ciphertext

// CHECK: func.func @chebyshev_consumes_levels
func.func @chebyshev_consumes_levels(%params: !params, %evaluator: !ckks_evaluator, %ct: !ct, %pt: !pt) -> (!ct, !ct) {
  %poly_eval = lattigo.ckks.new_polynomial_evaluator %params, %evaluator : (!params, !ckks_evaluator) -> !poly_eval
  // A degree-15 polynomial drops 4 levels.
  // CHECK: %[[CHEB:.*]] = lattigo.ckks.chebyshev
  %cheb = lattigo.ckks.chebyshev %poly_eval, %ct {coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], targetScale = 1073741824} : (!poly_eval, !ct) -> !ct
  // %cheb is dead after this point, but it is 4 levels below %ct, so the
  // add on the untouched %ct must not be given %cheb as its output buffer.
  // CHECK-NOT: lattigo.ckks.add %{{.*}}, %{{.*}}, %{{.*}}, %[[CHEB]]
  // CHECK: lattigo.ckks.add_new %{{.*}}, %{{.*}}, %{{.*}} :
  %high = lattigo.ckks.add_new %evaluator, %ct, %pt : (!ckks_evaluator, !ct, !pt) -> !ct
  return %cheb, %high : !ct, !ct
}
