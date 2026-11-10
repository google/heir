// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
#ring_1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
!poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=coeff>
!ntt_poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=eval>

module {
  // Covers: an scf.for iter_arg that needs both forms -- coeff to match its
  // loop result (consumed by to_tensor, coeff-only) and eval to feed the
  // eval-only MulOp in the loop body. Since the loop's iter_arg and result
  // share one materialized type, but each has its own demand driven by a
  // different consumer, the solver settles the whole loop (argument,
  // iter_arg, and result) on eval -- so the loop body needs zero
  // conversions, no matter how many times it runs -- and pays the one
  // unavoidable coeff conversion exactly once, right after the loop.
  // CHECK: func.func @for_iter_arg_needs_both_forms([[x:%.+]]: [[ntt_poly_ty_1:![^ ]+]],
  // CHECK-SAME: -> tensor<1024x[[RNS:![^ ]+]]> {
  // CHECK: [[r:%.+]] = scf.for {{.*}} iter_args([[acc:%.+]] = [[x]]) -> ([[ntt_poly_ty_1]]) {
  // CHECK: [[sq:%.+]] = polynomial.mul [[acc]], [[acc]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sq]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[rc:%.+]] = polynomial.intt [[r]] : [[ntt_poly_ty_1]]
  // CHECK: [[t:%.+]] = polynomial.to_tensor [[rc]] : [[poly_ty_1:![^ ]+]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t]] : tensor<1024x[[RNS]]>
  func.func @for_iter_arg_needs_both_forms(%x: !poly_ty_1, %lb: index, %ub: index, %step: index) -> tensor<1024x!rns.rns<!Zq0>> {
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> !poly_ty_1 {
      %sq = polynomial.mul %acc, %acc : !poly_ty_1
      scf.yield %sq : !poly_ty_1
    }
    %t = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t : tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: the same both-forms-needed iter_arg as above, but with a
  // statically-known trip count (4), so the solver isn't just falling back
  // to the "unknown trip count is assumed expensive" default -- it's
  // actually weighing "4 conversions inside the loop" against "1 conversion
  // outside it" and picking the latter because 1 < 4, confirming the
  // hoisting is genuinely cost-driven rather than an artifact of one
  // specific loop shape.
  // CHECK: func.func @for_iter_arg_static_trip_count([[x4:%.+]]: [[ntt_poly_ty_1]])
  // CHECK-SAME: -> tensor<1024x[[RNS]]> {
  // CHECK: [[r4:%.+]] = scf.for {{.*}} iter_args([[acc4:%.+]] = [[x4]]) -> ([[ntt_poly_ty_1]]) {
  // CHECK: [[sq4:%.+]] = polynomial.mul [[acc4]], [[acc4]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sq4]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[r4c:%.+]] = polynomial.intt [[r4]] : [[ntt_poly_ty_1]]
  // CHECK: [[t4:%.+]] = polynomial.to_tensor [[r4c]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t4]] : tensor<1024x[[RNS]]>
  func.func @for_iter_arg_static_trip_count(%x: !poly_ty_1) -> tensor<1024x!rns.rns<!Zq0>> {
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> !poly_ty_1 {
      %sq = polynomial.mul %acc, %acc : !poly_ty_1
      scf.yield %sq : !poly_ty_1
    }
    %t = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %t : tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: two independent polynomial iter_args in the same scf.for, each
  // getting its own, independently solved form. %acc0 needs coeff (feeds
  // to_tensor after the loop) and eval (feeds the eval-only mul in the loop
  // body); like the previous test, the solver settles the whole %acc0/%r0
  // slot on eval and pays one intt after the loop rather than converting on
  // every iteration. %acc1 merely passes through unused, so nothing
  // constrains its form at all -- any choice costs zero conversions, so the
  // solver's tie-break picks coeff.
  // CHECK: func.func @for_multiple_independent_iter_args([[a:%.+]]: [[ntt_poly_ty_1]], [[b:%.+]]: [[poly_ty_1]],
  // CHECK: [[loop:%.+]]:2 = scf.for {{.*}} iter_args([[acc0:%.+]] = [[a]], [[acc1:%.+]] = [[b]]) -> ([[ntt_poly_ty_1]], [[poly_ty_1]]) {
  // CHECK: [[sq0:%.+]] = polynomial.mul [[acc0]], [[acc0]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sq0]], [[acc1]] : [[ntt_poly_ty_1]], [[poly_ty_1]]
  // CHECK: }
  // CHECK: [[r0c:%.+]] = polynomial.intt [[loop]]#0 : [[ntt_poly_ty_1]]
  // CHECK: [[t:%.+]] = polynomial.to_tensor [[r0c]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[t]], [[loop]]#1 : tensor<1024x[[RNS]]>, [[poly_ty_1]]
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
  // (one per branch) to the same result. x needs coeff (to_tensor) and the
  // if's own result needs eval (the mul after it); scf.if executes each
  // branch exactly once, so it costs exactly the same (one conversion) to
  // land that conversion on x before the if or on the if's result after it.
  // The solver's tie-break prefers coeff mode, so x and the if's result both
  // stay coeff and the NTT lands on the result.
  // CHECK: func.func @if_yields_both_branches([[cond:%.+]]: i1, [[x2:%.+]]: [[poly_ty_1:![^ ]+]])
  // CHECK: [[r2:%.+]] = scf.if [[cond]] -> ([[poly_ty_1]]) {
  // CHECK: scf.yield [[x2]] : [[poly_ty_1]]
  // CHECK: } else {
  // CHECK: scf.yield [[x2]] : [[poly_ty_1]]
  // CHECK: }
  // CHECK: [[r2e:%.+]] = polynomial.ntt [[r2]] : [[poly_ty_1]]
  // CHECK: [[t2:%.+]] = polynomial.to_tensor [[x2]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: [[m2:%.+]] = polynomial.mul [[r2e]], [[r2e]] : [[ntt_poly_ty_1:![^ ]+]]
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
  // resolved form (see NTTSolver::addFloatingMode). Both the "before"
  // region's argument and this after-arg/result slot settle on eval: the
  // before-arg needs eval anyway (the mul), and forcing the after-arg/result
  // slot to coeff instead would mean converting on every iteration (inside
  // the "after" region) rather than paying the one coeff conversion
  // to_tensor needs exactly once, after the whole loop.
  // CHECK: func.func @while_loop([[x3:%.+]]: [[ntt_poly_ty_1:![^ ]+]], [[cond3:%.+]]: i1) -> tensor<1024x[[RNS:![^ ]+]]> {
  // CHECK: [[r3:%.+]] = scf.while ([[acc3:%.+]] = [[x3]]) : ([[ntt_poly_ty_1]]) -> [[ntt_poly_ty_1]] {
  // CHECK: [[sq3:%.+]] = polynomial.mul [[acc3]], [[acc3]] : [[ntt_poly_ty_1]]
  // CHECK: scf.condition({{.*}}) [[sq3]] : [[ntt_poly_ty_1]]
  // CHECK: } do {
  // CHECK: ^bb0([[after3:%.+]]: [[ntt_poly_ty_1]]):
  // CHECK: scf.yield [[after3]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[r3c:%.+]] = polynomial.intt [[r3]] : [[ntt_poly_ty_1]]
  // CHECK: [[t3:%.+]] = polynomial.to_tensor [[r3c]] : [[poly_ty_1:![^ ]+]] -> tensor<1024x[[RNS]]>
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

  // Covers: an NTT hoisted out of a loop. %x comes from a coeff-only op
  // (from_tensor), the loop body needs eval (mul), and the loop result
  // needs coeff (to_tensor). Without per-iteration weighting this is a tie:
  // keeping the loop in eval costs an NTT on %x before the loop plus an INTT
  // after it (2), while keeping it in coeff costs an NTT on %acc plus an INTT
  // on %sq inside the body (also 2, unweighted). Only weighting the in-loop
  // conversions by the trip count makes the hoisted version strictly cheaper
  // (2 vs. 2 * kUnknownLoopIterations), so this checks the weighting, not
  // just the loop structure.
  // CHECK: func.func @for_hoists_ntt_out_of_loop([[t5:%.+]]: tensor<1024x[[RNS]]>,
  // CHECK: [[x5:%.+]] = polynomial.from_tensor [[t5]] : tensor<1024x[[RNS]]> -> [[poly_ty_1]]
  // CHECK: [[x5e:%.+]] = polynomial.ntt [[x5]] : [[poly_ty_1]]
  // CHECK: [[r5:%.+]] = scf.for {{.*}} iter_args([[acc5:%.+]] = [[x5e]]) -> ([[ntt_poly_ty_1]]) {
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: [[sq5:%.+]] = polynomial.mul [[acc5]], [[acc5]] : [[ntt_poly_ty_1]]
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: scf.yield [[sq5]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[r5c:%.+]] = polynomial.intt [[r5]] : [[ntt_poly_ty_1]]
  // CHECK: [[out5:%.+]] = polynomial.to_tensor [[r5c]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[out5]] : tensor<1024x[[RNS]]>
  func.func @for_hoists_ntt_out_of_loop(%t: tensor<1024x!rns.rns<!Zq0>>, %lb: index, %ub: index, %step: index) -> tensor<1024x!rns.rns<!Zq0>> {
    %x = polynomial.from_tensor %t : tensor<1024x!rns.rns<!Zq0>> -> !poly_ty_1
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> !poly_ty_1 {
      %sq = polynomial.mul %acc, %acc : !poly_ty_1
      scf.yield %sq : !poly_ty_1
    }
    %out = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %out : tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: nested loops, where a conversion's weight is the *product* of the
  // enclosing trip counts (outer N = 5, inner M = 2). The outer body needs
  // coeff (monic_monomial_mul) and the inner body needs eval (mul). The
  // cheapest plan keeps the outer loop in coeff and the inner loop in eval,
  // paying one NTT (on %m) and one INTT (on the inner result), both in the
  // outer body: 2 * N = 10. Running the inner loop in coeff instead puts an
  // NTT and an INTT inside the inner body: 2 * N * M = 20. If the inner
  // weight were just M (no product), that would cost 2 * M = 4 and wrongly
  // win, so this pins down the multiplication. Running the outer loop in
  // eval costs 2 * N + 2 = 12 (conversions on %a and %m in the outer body,
  // plus one on %x before and one on the result after the loop).
  // CHECK: func.func @nested_for_trip_counts_multiply([[t6:%.+]]: tensor<1024x[[RNS]]>, [[k6:%.+]]: index)
  // CHECK: [[x6:%.+]] = polynomial.from_tensor [[t6]] : tensor<1024x[[RNS]]> -> [[poly_ty_1]]
  // CHECK-NOT: polynomial.ntt
  // CHECK: [[r6:%.+]] = scf.for {{.*}} iter_args([[a6:%.+]] = [[x6]]) -> ([[poly_ty_1]]) {
  // CHECK: [[m6:%.+]] = polynomial.monic_monomial_mul [[a6]], [[k6]] : ([[poly_ty_1]], index) -> [[poly_ty_1]]
  // CHECK: [[m6e:%.+]] = polynomial.ntt [[m6]] : [[poly_ty_1]]
  // CHECK: [[ri6:%.+]] = scf.for {{.*}} iter_args([[b6:%.+]] = [[m6e]]) -> ([[ntt_poly_ty_1]]) {
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: [[sq6:%.+]] = polynomial.mul [[b6]], [[b6]] : [[ntt_poly_ty_1]]
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: scf.yield [[sq6]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: [[ri6c:%.+]] = polynomial.intt [[ri6]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[ri6c]] : [[poly_ty_1]]
  // CHECK: }
  // CHECK-NOT: polynomial.intt
  // CHECK: [[out6:%.+]] = polynomial.to_tensor [[r6]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[out6]] : tensor<1024x[[RNS]]>
  func.func @nested_for_trip_counts_multiply(%t: tensor<1024x!rns.rns<!Zq0>>, %k: index) -> tensor<1024x!rns.rns<!Zq0>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %x = polynomial.from_tensor %t : tensor<1024x!rns.rns<!Zq0>> -> !poly_ty_1
    %r = scf.for %i = %c0 to %c5 step %c1 iter_args(%a = %x) -> !poly_ty_1 {
      %m = polynomial.monic_monomial_mul %a, %k : (!poly_ty_1, index) -> !poly_ty_1
      %ri = scf.for %j = %c0 to %c2 step %c1 iter_args(%b = %m) -> !poly_ty_1 {
        %sq = polynomial.mul %b, %b : !poly_ty_1
        scf.yield %sq : !poly_ty_1
      }
      scf.yield %ri : !poly_ty_1
    }
    %out = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %out : tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: an ordinary (non-region-branch) op inside a loop body whose
  // result conversion is weighted by the trip count. %s = add %w0, %w1
  // (either-form) is computed inside the loop from two loop-invariant,
  // coeff-only values, and feeds an eval-only mul. Running the add in coeff
  // mode needs one NTT on %s inside the body (1 unweighted, but
  // kUnknownLoopIterations weighted); running it in eval mode needs one NTT
  // on each of %w0 and %w1 at their definitions outside the loop (2). The
  // unweighted cost would favor the in-loop NTT, so this checks that %s's
  // conversion is charged per iteration.
  // CHECK: func.func @for_body_op_conversion_weighted([[x7:%.+]]: [[ntt_poly_ty_1]], [[t7a:%.+]]: tensor<1024x[[RNS]]>, [[t7b:%.+]]: tensor<1024x[[RNS]]>,
  // CHECK: [[w7a:%.+]] = polynomial.from_tensor [[t7a]] : tensor<1024x[[RNS]]> -> [[poly_ty_1]]
  // CHECK: [[w7ae:%.+]] = polynomial.ntt [[w7a]] : [[poly_ty_1]]
  // CHECK: [[w7b:%.+]] = polynomial.from_tensor [[t7b]] : tensor<1024x[[RNS]]> -> [[poly_ty_1]]
  // CHECK: [[w7be:%.+]] = polynomial.ntt [[w7b]] : [[poly_ty_1]]
  // CHECK: [[r7:%.+]] = scf.for {{.*}} iter_args([[acc7:%.+]] = [[x7]]) -> ([[ntt_poly_ty_1]]) {
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: [[s7:%.+]] = polynomial.add [[w7ae]], [[w7be]] : [[ntt_poly_ty_1]]
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: [[sq7:%.+]] = polynomial.mul [[acc7]], [[s7]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[sq7]] : [[ntt_poly_ty_1]]
  // CHECK: }
  // CHECK: return [[r7]] : [[ntt_poly_ty_1]]
  func.func @for_body_op_conversion_weighted(%x: !poly_ty_1, %t0: tensor<1024x!rns.rns<!Zq0>>, %t1: tensor<1024x!rns.rns<!Zq0>>, %lb: index, %ub: index, %step: index) -> !poly_ty_1 {
    %w0 = polynomial.from_tensor %t0 : tensor<1024x!rns.rns<!Zq0>> -> !poly_ty_1
    %w1 = polynomial.from_tensor %t1 : tensor<1024x!rns.rns<!Zq0>> -> !poly_ty_1
    %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %x) -> !poly_ty_1 {
      %s = polynomial.add %w0, %w1 : !poly_ty_1
      %sq = polynomial.mul %acc, %s : !poly_ty_1
      scf.yield %sq : !poly_ty_1
    }
    return %r : !poly_ty_1
  }

  // Covers: scf.if with real work inside each branch, where the branches
  // produce different native forms (mul -> eval, monic_monomial_mul -> coeff)
  // for the same result, and the result's consumer (to_tensor) needs coeff.
  // The result settles on coeff, so the only conversion is an INTT inside
  // the "then" branch, right after the mul and before its scf.yield (cost 1).
  // Settling on eval would need an NTT in the "else" branch plus an INTT
  // after the if (cost 2). %x and %y each need only one form, so the
  // function signature has no tie to break either.
  // CHECK: func.func @if_converts_inside_branch([[cond8:%.+]]: i1, [[x8:%.+]]: [[ntt_poly_ty_1]], [[y8:%.+]]: [[poly_ty_1]], [[k8:%.+]]: index)
  // CHECK: [[r8:%.+]] = scf.if [[cond8]] -> ([[poly_ty_1]]) {
  // CHECK: [[m8:%.+]] = polynomial.mul [[x8]], [[x8]] : [[ntt_poly_ty_1]]
  // CHECK: [[m8c:%.+]] = polynomial.intt [[m8]] : [[ntt_poly_ty_1]]
  // CHECK: scf.yield [[m8c]] : [[poly_ty_1]]
  // CHECK: } else {
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: [[s8:%.+]] = polynomial.monic_monomial_mul [[y8]], [[k8]] : ([[poly_ty_1]], index) -> [[poly_ty_1]]
  // CHECK: scf.yield [[s8]] : [[poly_ty_1]]
  // CHECK: }
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: [[out8:%.+]] = polynomial.to_tensor [[r8]] : [[poly_ty_1]] -> tensor<1024x[[RNS]]>
  // CHECK: return [[out8]] : tensor<1024x[[RNS]]>
  func.func @if_converts_inside_branch(%cond: i1, %x: !poly_ty_1, %y: !poly_ty_1, %k: index) -> tensor<1024x!rns.rns<!Zq0>> {
    %r = scf.if %cond -> !poly_ty_1 {
      %m = polynomial.mul %x, %x : !poly_ty_1
      scf.yield %m : !poly_ty_1
    } else {
      %s = polynomial.monic_monomial_mul %y, %k : (!poly_ty_1, index) -> !poly_ty_1
      scf.yield %s : !poly_ty_1
    }
    %out = polynomial.to_tensor %r : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    return %out : tensor<1024x!rns.rns<!Zq0>>
  }
}
