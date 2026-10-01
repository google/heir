// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
#ring_1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
!poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=coeff>
!ntt_poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=eval>

module {
  // Covers: an scf.for iter_arg that needs both forms -- coeff to match its
  // entry operand/loop result (consumed by to_tensor, coeff-only) and eval to
  // feed the eval-only MulOp in the loop body. The NTT/INTT pair is hoisted
  // to exactly one occurrence per iteration, inside the loop body around the
  // mul, with no conversions needed on the function argument or the loop
  // result themselves.
  // CHECK: func.func @for_iter_arg_needs_both_forms([[x:%.+]]: [[poly_ty_1:![^ ]+]],
  // CHECK-SAME: -> tensor<1024x[[RNS:![^ ]+]]> {
  // CHECK: [[r:%.+]] = scf.for {{.*}} iter_args([[acc:%.+]] = [[x]]) -> ([[poly_ty_1]]) {
  // CHECK: [[acce:%.+]] = polynomial.ntt [[acc]] : [[poly_ty_1]]
  // CHECK: [[sq:%.+]] = polynomial.mul [[acce]], [[acce]] : [[ntt_poly_ty_1:![^ ]+]]
  // CHECK: [[sqc:%.+]] = polynomial.intt [[sq]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sqc]] : [[poly_ty_1]]
  // CHECK: [[t:%.+]] = polynomial.to_tensor [[r]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t]] : tensor<1024x[[RNS]]>
  func.func @for_iter_arg_needs_both_forms(%x: !poly_ty_1, %lb: index, %ub: index, %step: index) -> tensor<1024x!rns.rns<!Zq0>> {
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> !poly_ty_1 {
      %sq = polynomial.mul %acc, %acc : !poly_ty_1
      scf.yield %sq : !poly_ty_1
    }
    %t = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t : tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: two independent polynomial iter_args in the same scf.for, each
  // getting its own, independently solved form. %acc0 needs coeff (feeds
  // to_tensor after the loop) and gets NTT/INTT'd locally around the
  // eval-only mul, exactly as above. %acc1 merely passes through unused, so
  // it's free to settle on eval form throughout with no conversions at all.
  // CHECK: func.func @for_multiple_independent_iter_args([[a:%.+]]: [[poly_ty_1]], [[b:%.+]]: [[ntt_poly_ty_1]],
  // CHECK: [[loop:%.+]]:2 = scf.for {{.*}} iter_args([[acc0:%.+]] = [[a]], [[acc1:%.+]] = [[b]]) -> ([[poly_ty_1]], [[ntt_poly_ty_1]]) {
  // CHECK: [[acc0e:%.+]] = polynomial.ntt [[acc0]] : [[poly_ty_1]]
  // CHECK: [[sq0:%.+]] = polynomial.mul [[acc0e]], [[acc0e]] : [[ntt_poly_ty_1]]
  // CHECK: [[sq0c:%.+]] = polynomial.intt [[sq0]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sq0c]], [[acc1]] : [[poly_ty_1]], [[ntt_poly_ty_1]]
  // CHECK: [[t:%.+]] = polynomial.to_tensor [[loop]]#0 : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t]], [[loop]]#1 : tensor<1024x[[RNS]]>, [[ntt_poly_ty_1]]
  func.func @for_multiple_independent_iter_args(%a: !poly_ty_1, %b: !poly_ty_1, %lb: index, %ub: index, %step: index) -> (tensor<1024x!rns.rns<!Zq0>>, !poly_ty_1) {
    %r0, %r1 = scf.for %i = %lb to %ub step %step iter_args(%acc0 = %a, %acc1 = %b) -> (!poly_ty_1, !poly_ty_1) {
      %sq0 = polynomial.mul %acc0, %acc0 : !poly_ty_1
      scf.yield %sq0, %acc1 : !poly_ty_1, !poly_ty_1
    }
    %t = polynomial.to_tensor %r0 : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t, %r1 : tensor<1024x!rns.rns<!Zq0>>, !poly_ty_1
  }

  // Covers: a tensor<poly> loop-carried value. Since the result is returned
  // directly with no coeff-only consumer, the whole loop (argument, iter_arg,
  // and result) settles on eval form for free, with no conversions inserted.
  // CHECK: func.func @for_tensor_iter_arg([[xt:%.+]]: tensor<2x[[ntt_poly_ty_1]]>,
  // CHECK: [[rt:%.+]] = scf.for {{.*}} iter_args([[acct:%.+]] = [[xt]]) -> (tensor<2x[[ntt_poly_ty_1]]>) {
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: [[sqt:%.+]] = polynomial.mul [[acct]], [[acct]] : tensor<2x[[ntt_poly_ty_1]]>
  // CHECK: scf.yield [[sqt]] : tensor<2x[[ntt_poly_ty_1]]>
  // CHECK: return [[rt]] : tensor<2x[[ntt_poly_ty_1]]>
  func.func @for_tensor_iter_arg(%x: tensor<2x!poly_ty_1>, %lb: index, %ub: index, %step: index) -> tensor<2x!poly_ty_1> {
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> tensor<2x!poly_ty_1> {
      %sq = polynomial.mul %acc, %acc : tensor<2x!poly_ty_1>
      scf.yield %sq : tensor<2x!poly_ty_1>
    }
    return %r : tensor<2x!poly_ty_1>
  }

  // Covers: scf.if yielding the same polynomial from both branches, with a
  // coeff-only consumer of the original value and an eval-only consumer of
  // the if's result. Since scf.if's regions take no block arguments, this
  // exercises only the "value returned to the parent" successor-input case
  // (not entry/backedge edges), forwarded from two different scf.yield ops
  // (one per branch) to the same result. x needs both forms (coeff for
  // to_tensor, eval to feed the if); the if's own result only ever needs
  // eval, so the one unavoidable conversion lands on x rather than on it.
  // CHECK: func.func @if_yields_both_branches([[cond:%.+]]: i1, [[x2:%.+]]: [[poly_ty_1]])
  // CHECK: [[x2e:%.+]] = polynomial.ntt [[x2]] : [[poly_ty_1]]
  // CHECK: [[r2:%.+]] = scf.if [[cond]] -> ([[ntt_poly_ty_1]]) {
  // CHECK: scf.yield [[x2e]] : [[ntt_poly_ty_1]]
  // CHECK: } else {
  // CHECK: scf.yield [[x2e]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[t2:%.+]] = polynomial.to_tensor [[x2]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: [[m2:%.+]] = polynomial.mul [[r2]], [[r2]] : [[ntt_poly_ty_1]]
  // CHECK: return [[t2]], [[m2]] : tensor<1024x[[RNS]]>, [[ntt_poly_ty_1]]
  func.func @if_yields_both_branches(%cond: i1, %x: !poly_ty_1) -> (tensor<1024x!rns.rns<!Zq0>>, !poly_ty_1) {
    %r = scf.if %cond -> !poly_ty_1 {
      scf.yield %x : !poly_ty_1
    } else {
      scf.yield %x : !poly_ty_1
    }
    %t = polynomial.to_tensor %x : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    %m = polynomial.mul %r, %r : !poly_ty_1
    return %t, %m : tensor<1024x!rns.rns<!Zq0>>, !poly_ty_1
  }

  // Covers: scf.while, whose scf.condition op forwards the very same
  // operand to two different successor inputs at once -- the "after"
  // region's block argument (if the loop continues) and the scf.while op's
  // own result (if it exits) -- exercising the case where a single physical
  // operand fans out to multiple successor inputs that must share one
  // resolved form (see NTTSolver::equateNativeForm). to_tensor forces the
  // while's result (and, by the fan-out above, the "after" region's
  // argument) to coeff. The "before" region's argument has no such
  // constraint, so it's free to settle on eval directly to feed the
  // eval-only mul with no local conversion; the fan-out's coeff requirement
  // is instead satisfied by a single intt before scf.condition, and the
  // "after" region converts back to eval before looping around.
  // CHECK: func.func @while_loop([[x3:%.+]]: [[ntt_poly_ty_1]], [[cond3:%.+]]: i1) -> tensor<1024x[[RNS]]> {
  // CHECK: [[r3:%.+]] = scf.while ([[acc3:%.+]] = [[x3]]) : ([[ntt_poly_ty_1]]) -> [[poly_ty_1]] {
  // CHECK: [[sq3:%.+]] = polynomial.mul [[acc3]], [[acc3]] : [[ntt_poly_ty_1]]
  // CHECK: [[sq3c:%.+]] = polynomial.intt [[sq3]] : [[ntt_poly_ty_1]]
  // CHECK: scf.condition({{.*}}) [[sq3c]] : [[poly_ty_1]]
  // CHECK: } do {
  // CHECK: ^bb0([[after3:%.+]]: [[poly_ty_1]]):
  // CHECK: [[after3e:%.+]] = polynomial.ntt [[after3]] : [[poly_ty_1]]
  // CHECK: scf.yield [[after3e]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[t3:%.+]] = polynomial.to_tensor [[r3]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t3]] : tensor<1024x[[RNS]]>
  func.func @while_loop(%x: !poly_ty_1, %cond0: i1) -> tensor<1024x!rns.rns<!Zq0>> {
    %r = scf.while (%acc = %x) : (!poly_ty_1) -> !poly_ty_1 {
      %sq = polynomial.mul %acc, %acc : !poly_ty_1
      %c = arith.constant true
      scf.condition(%c) %sq : !poly_ty_1
    } do {
    ^bb0(%arg: !poly_ty_1):
      scf.yield %arg : !poly_ty_1
    }
    %t = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t : tensor<1024x!rns.rns<!Zq0>>
  }
}
