// RUN: heir-opt --convert-if-to-select --split-input-file --verify-diagnostics %s

func.func private @printer(%inp: tensor<16xi16>) -> ()

func.func @impure_operation(%inp: !secret.secret<tensor<16xi16>>, %cond: i1) -> !secret.secret<tensor<16xi16>> {
  %0 = secret.generic ins(%inp, %cond : !secret.secret<tensor<16xi16>>, i1) {
  ^bb0(%secret_inp: tensor<16xi16>, %copy_cond: i1):
// expected-error@below {{Can't convert scf.if to arith.select operation. If-operation contains code that can't be safely hoisted on line }}
    %1 = scf.if %copy_cond -> (tensor<16xi16>) {
      %2 = arith.addi %secret_inp, %secret_inp : tensor<16xi16>
      func.call @printer(%2) : (tensor<16xi16>) -> ()
      scf.yield %2 : tensor<16xi16>
    } else {
      scf.yield %secret_inp : tensor<16xi16>
    }
    secret.yield %1 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  return %0 : !secret.secret<tensor<16xi16>>
}

// -----

func.func @non_speculative_code(%inp: !secret.secret<i16>, %divisor: !secret.secret<i16>) -> !secret.secret<i16> {
  %0 = secret.generic ins(%inp, %divisor : !secret.secret<i16>, !secret.secret<i16>) {
  ^bb0(%secret_inp: i16, %secret_divisor: i16):
    %zero = arith.constant 0 : i16
    %secret_cond = arith.cmpi eq, %zero, %secret_divisor : i16
// expected-error@below {{Can't convert scf.if to arith.select operation. If-operation contains code that can't be safely hoisted on line }}
    %1 = scf.if %secret_cond -> (i16) {
      %2 = arith.divui %secret_inp, %secret_divisor : i16 // non-pure
      scf.yield %2 : i16
    } else {
      scf.yield %secret_inp : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}

// -----

func.func @conditionally_speculative_code(%inp: !secret.secret<i16>, %cond :!secret.secret<i1>) -> !secret.secret<i16> {
  %divisor = arith.constant 0 : i16
  %0 = secret.generic ins(%inp, %cond : !secret.secret<i16>, !secret.secret<i1>) {
  ^bb0(%secret_inp: i16, %secret_cond: i1):
// expected-error@below {{Can't convert scf.if to arith.select operation. If-operation contains code that can't be safely hoisted on line }}
    %1 = scf.if %secret_cond -> (i16) {
      %2 = arith.divui %secret_inp, %divisor : i16
      scf.yield %2 : i16
    } else {
      scf.yield %secret_inp : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
