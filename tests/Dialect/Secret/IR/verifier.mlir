// RUN: heir-opt --verify-diagnostics --split-input-file %s

func.func @test_cleartext_type_mismatch(%value: i32, %c1: i32) {
  // expected-error@+1 {{If the operand is not secret, it must be the same type as}}
  %Z = secret.generic
    ins(%value, %c1 : i32, i32) {
    ^bb0(%clear_value: i64, %clear_c1: i64):
      %0 = arith.addi %clear_value, %clear_c1 : i64
      secret.yield %0 : i64
    } -> (!secret.secret<i64>)
  return
}

// -----

func.func @test_secret_type_mismatch(%value: !secret.secret<i32>, %c1: i32) {
  // expected-error@+1 {{Type mismatch between block argument 0 of type 'i64' and generic operand of type '!secret.secret<i32>'}}
  %Z = secret.generic
    ins(%value, %c1 : !secret.secret<i32>, i32) {
    ^bb0(%clear_value: i64, %clear_c1: i32):
      %0 = arith.trunci %clear_value : i64 to i32
      %1 = arith.addi %0, %clear_c1 : i32
      secret.yield %1 : i32
    } -> (!secret.secret<i32>)
  return
}

// -----

func.func @ensure_yield_inside_generic(%value: !secret.secret<i32>) {
  // expected-error@+1 {{expects parent op 'secret.generic'}}
  secret.yield %value : !secret.secret<i32>
  return
}

// -----

func.func @test_yield_type_agrees_with_generic(%value: !secret.secret<i32>) {
  %Z = secret.generic
    ins(%value : !secret.secret<i32>) {
    ^bb0(%clear_value: i32):
      %1 = arith.addi %clear_value, %clear_value : i32
      %2 = arith.extui %1 : i32 to i64
      // expected-error@+1 {{If a yield op returns types T, S, ..., then the enclosing generic op must have result types secret.secret<T>, secret.secret<S>, ... But this yield op has operand types: 'i64'; while the enclosing generic op has result types: '!secret.secret<i32>'}}
      secret.yield %2 : i64
    } -> (!secret.secret<i32>)
  return
}
