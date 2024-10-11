// RUN: heir-opt --convert-secret-while-to-static-for --split-input-file --verify-diagnostics %s

func.func @while_loop_without_max_iter(%input: !secret.secret<i16>) -> !secret.secret<i16> {
  %c100 = arith.constant 100 : i16
  %c20 = arith.constant 20 : i16
  %0 = secret.generic ins(%input : !secret.secret<i16>) {
  ^bb0(%arg1: i16):
    // expected-warning@+1 {{Cannot convert secret scf.while to static affine.for since a static maximum iteration attribute (`max_iter`) has not been provided on the scf.while op:}}
    %1 = scf.while (%arg2 = %arg1) : (i16) -> i16 {
      %3 = arith.cmpi slt, %arg2, %c100 : i16
      scf.condition(%3) %arg2 : i16
    } do {
    ^bb0(%arg2: i16):
      %2 = arith.muli %arg2, %arg2 : i16
      scf.yield %2 : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}

// ----

// CHECK-LABEL: @do_while_not_supported
func.func @do_while_not_supported(%input: !secret.secret<i16>) -> !secret.secret<i16> {
  %c100 = arith.constant 100 : i16
  %c20 = arith.constant 20 : i16
  %0 = secret.generic ins(%input : !secret.secret<i16>) {
  ^bb0(%arg1: i16):
    // expected-warning@+1 {{Current loop transformation has no support for do-while loops:}}
    %1 = scf.while (%arg2 = %arg1) : (i16) -> i16 {
      %3 = arith.cmpi sgt, %arg2, %c100 : i16
      %arg4  = arith.muli %arg2, %c100 : i16
      scf.condition(%3) %arg4 : i16
    } do {
    ^bb0(%arg4: i16):
      %2 = arith.muli %arg4, %arg4 : i16
      scf.yield %2 : i16
    } attributes {max_iter = 16 : i64}
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
