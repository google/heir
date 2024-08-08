// RUN: heir-opt --convert-if-to-select  %s | FileCheck %s

// CHECK-LABEL: @secret_condition_with_non_secret_int
func.func @secret_condition_with_non_secret_int(%inp: i16, %cond: !secret.secret<i1>) -> !secret.secret<i16> {
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[INP:.*]], %[[COND:.*]] : [[T:.*]], !secret.secret<i1>) {
  // CHECK-NEXT:   ^[[bb0:.*]](%[[CPY_INP:.*]]: [[T]], %[[SCRT_COND:.*]]: i1):
  // CHECK-NEXT:     %[[ADD:.*]] = arith.addi %[[CPY_INP]], %[[CPY_INP]] : [[T]]
  // CHECK-NEXT:     %[[SEL:.*]] = arith.select %[[SCRT_COND]], %[[ADD]], %[[CPY_INP]] : [[T]]
  // CHECK-NEXT:     secret.yield %[[SEL]] : [[T]]
  // CHECK-NEXT:   } -> !secret.secret<[[T]]>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<[[T]]>
  %0 = secret.generic ins(%inp, %cond : i16, !secret.secret<i1>) {
  ^bb0(%copy_inp: i16, %secret_cond: i1):
    %1 = scf.if %secret_cond -> (i16) {
      %2 = arith.addi %copy_inp, %copy_inp : i16
      scf.yield %2 : i16
    } else {
      scf.yield %copy_inp : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}


// CHECK-LABEL: @secret_condition_with_secret_int
func.func @secret_condition_with_secret_int(%inp: !secret.secret<i16>, %cond: !secret.secret<i1>) -> !secret.secret<i16> {
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[INP:.*]], %[[COND:.*]] : !secret.secret<[[T:.*]]>, !secret.secret<i1>) {
  // CHECK-NEXT:   ^[[bb0:.*]](%[[SCRT_INP:.*]]: [[T]], %[[SCRT_COND:.*]]: i1):
  // CHECK-NEXT:     %[[ADD:.*]] = arith.addi %[[SCRT_INP]], %[[SCRT_INP]] : [[T]]
  // CHECK-NEXT:     %[[SEL:.*]] = arith.select %[[SCRT_COND]], %[[ADD]], %[[SCRT_INP]] : [[T]]
  // CHECK-NEXT:     secret.yield %[[SEL]] : [[T]]
  // CHECK-NEXT:  } -> !secret.secret<[[T]]>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<[[T]]>
  %0 = secret.generic ins(%inp, %cond : !secret.secret<i16>, !secret.secret<i1>) {
  ^bb0(%secret_inp: i16, %secret_cond: i1):
    %1 = scf.if %secret_cond -> (i16) {
      %2 = arith.addi %secret_inp, %secret_inp : i16
      scf.yield %2 : i16
    } else {
      scf.yield %secret_inp : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}


// CHECK-LABEL: @secret_condition_with_secret_int_and_multiple_yields
func.func @secret_condition_with_secret_int_and_multiple_yields(%inp: !secret.secret<i16>, %cond: !secret.secret<i1>) -> !secret.secret<i16> {
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[INP:.*]], %[[COND:.*]] : !secret.secret<[[T:.*]]>, !secret.secret<i1>) {
  // CHECK-NEXT:  ^[[bb0:.*]](%[[SCRT_INP:.*]]: [[T]], %[[SCRT_COND:.*]]: i1):
  // CHECK-NEXT:    %[[ADD1:.*]] = arith.addi %[[SCRT_INP]], %[[SCRT_INP]] : [[T]]
  // CHECK-NEXT:    %[[MUL:.*]] = arith.muli %[[SCRT_INP]], %[[ADD1]] : [[T]]
  // CHECK-NEXT:    %[[SEL1:.*]] = arith.select %[[SCRT_COND]], %[[ADD1]], %[[SCRT_INP]]  : [[T]]
  // CHECK-NEXT:    %[[SEL2:.*]] = arith.select %[[SCRT_COND]], %[[MUL]], %[[SCRT_INP]]  : [[T]]
  // CHECK-NEXT:    %[[ADD2:.*]] = arith.addi %[[SEL1]], %[[SEL2]] : [[T]]
  // CHECK-NEXT:    secret.yield %[[ADD2]] : [[T]]
  // CHECK-NEXT:  } -> !secret.secret<[[T]]>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<[[T]]>
  %0 = secret.generic ins(%inp, %cond : !secret.secret<i16>, !secret.secret<i1>) {
    ^bb0(%secret_inp: i16, %secret_cond: i1):
      %1, %3 = scf.if %secret_cond -> (i16, i16) {
        %2 = arith.addi %secret_inp, %secret_inp : i16
        %4 = arith.muli %secret_inp, %2 : i16
        scf.yield %2, %4 : i16, i16
      } else {
        scf.yield %secret_inp, %secret_inp : i16, i16
      }
      %5 = arith.addi %1, %3 : i16
      secret.yield %5 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
  }


// CHECK-LABEL: @secret_condition_with_secret_tensor
func.func @secret_condition_with_secret_tensor(%inp: !secret.secret<tensor<16xi16>>, %cond: !secret.secret<i1>) -> !secret.secret<tensor<16xi16>> {
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[INP:.*]], %[[COND:.*]] : !secret.secret<tensor<[[T:.*]]>>, !secret.secret<i1>)
  // CHECK-NEXT:  ^[[bb0:.*]](%[[SCRT_INP:.*]]: tensor<[[T:.*]]>, %[[SCRT_COND:.*]]: i1):
  // CHECK-NEXT:    %[[ADD:.*]] = arith.addi %[[SCRT_INP]], %[[SCRT_INP]] : tensor<[[T]]>
  // CHECK-NEXT:    %[[MUL:.*]] = arith.muli %[[SCRT_INP]], %[[SCRT_INP]] : tensor<[[T]]>
  // CHECK-NEXT:    %[[SEL:.*]] = arith.select %[[SCRT_COND]], %[[ADD]], %[[MUL]] : tensor<[[T]]>
  // CHECK-NEXT:    secret.yield %[[SEL]] : tensor<[[T]]>
  // CHECK-NEXT:  } -> !secret.secret<tensor<[[T]]>>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<tensor<[[T]]>>
  %0 = secret.generic ins(%inp, %cond : !secret.secret<tensor<16xi16>>, !secret.secret<i1>) {
  ^bb0(%secret_inp: tensor<16xi16>, %secret_cond: i1):
    %1 = scf.if %secret_cond -> (tensor<16xi16>) {
      %2 = arith.addi %secret_inp, %secret_inp : tensor<16xi16>
      scf.yield %2 : tensor<16xi16>
    } else {
      %2 = arith.muli %secret_inp, %secret_inp : tensor<16xi16>
      scf.yield %2 : tensor<16xi16>
    }
    secret.yield %1 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  return %0 : !secret.secret<tensor<16xi16>>
}


// CHECK-LABEL: @secret_condition_with_secret_vector
func.func @secret_condition_with_secret_vector(%inp: !secret.secret<vector<4xf32>>, %cond: !secret.secret<i1>) -> !secret.secret<vector<4xf32>> {
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[INP:.*]], %[[COND:.*]] : !secret.secret<vector<[[T:.*]]>>, !secret.secret<i1>) {
  // CHECK-NEXT:  ^[[bb0:.*]](%[[SCRT_INP:.*]]: vector<[[T]]>, %[[SCRT_COND:.*]]: i1):
  // CHECK-NEXT:   %[[ADD:.*]] = arith.addf %[[SCRT_INP]], %[[SCRT_INP]] : vector<[[T]]>
  // CHECK-NEXT:   %[[SEL:.*]] = arith.select %[[SCRT_COND]], %[[ADD]], %[[SCRT_INP]] : vector<[[T]]>
  // CHECK-NEXT:   secret.yield %[[SEL]] : vector<[[T]]>
  // CHECK-NEXT:  } -> !secret.secret<vector<[[T]]>>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<vector<[[T]]>>
  %0 = secret.generic ins(%inp, %cond : !secret.secret<vector<4xf32>>, !secret.secret<i1>) {
  ^bb0(%secret_inp: vector<4xf32>, %secret_cond: i1):
    %1 = scf.if %secret_cond -> (vector<4xf32>) {
      %2 = arith.addf %secret_inp, %secret_inp : vector<4xf32>
      scf.yield %2 : vector<4xf32>
    } else {
      scf.yield %secret_inp : vector<4xf32>
    }
    secret.yield %1 : vector<4xf32>
  } -> !secret.secret<vector<4xf32>>
  return %0 : !secret.secret<vector<4xf32>>
}

// CHECK-LABEL: @tainted_condition
func.func @tainted_condition(%inp: !secret.secret<i16>) -> !secret.secret<i16>{
  // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : [[T:.*]]
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[INP:.*]] : !secret.secret<[[T]]>) {
  // CHECK-NEXT:  ^[[bb0:.*]](%[[SCRT_INP:.*]]: [[T]]):
  // CHECK-NEXT:    %[[CMP:.*]] = arith.cmpi eq, %[[SCRT_INP]], %[[ZERO]]  : [[T]]
  // CHECK-NEXT:    %[[ADD:.*]] = arith.addi %[[SCRT_INP]], %[[SCRT_INP]] : [[T]]
  // CHECK-NEXT:    %[[SEL:.*]] = arith.select %[[CMP]], %[[ADD]], %[[SCRT_INP]] : [[T]]
  // CHECK-NEXT:    secret.yield %[[SEL]] : [[T]]
  // CHECK-NEXT:  } -> !secret.secret<[[T]]>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<[[T]]>
  %0 = arith.constant 0 : i16
  %1 = secret.generic ins(%inp: !secret.secret<i16>) {
  ^bb0(%secret_inp: i16):
    %2 = arith.cmpi eq, %secret_inp, %0 : i16
    %3 = scf.if %2  -> (i16) {
      %4 = arith.addi %secret_inp, %secret_inp : i16
      scf.yield %4 : i16
    } else {
        scf.yield %secret_inp : i16
    }
    secret.yield %3 : i16
    } -> !secret.secret<i16>

    return %1 : !secret.secret<i16>
}

// CHECK-LABEL: @speculatable_divison
func.func @speculatable_divison(%inp: !secret.secret<i16>, %cond :!secret.secret<i1>) -> !secret.secret<i16> {
  // CHECK-NEXT: %[[DIVISOR:.*]] = arith.constant 2 : i16
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[INP:.*]], %[[COND:.*]] : !secret.secret<[[T:.*]]>, !secret.secret<i1>) {
  // CHECK-NEXT:  ^[[bb0:.*]](%[[SCRT_INP:.*]]: [[T]], %[[SCRT_COND:.*]]: i1):
  // CHECK-NEXT:    %[[DIV:.*]] = arith.divui %[[SCRT_INP]], %[[DIVISOR]] : [[T]]
  // CHECK-NEXT:    %[[SEL:.*]] = arith.select %[[SCRT_COND]], %[[DIV]], %[[SCRT_INP]] : [[T]]
  // CHECK-NEXT: secret.yield %[[SEL]] : [[T]]
  // CHECK-NEXT:  } -> !secret.secret<[[T]]>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<[[T]]>
  %divisor = arith.constant 2 : i16
  %0 = secret.generic ins(%inp, %cond : !secret.secret<i16>, !secret.secret<i1>) {
  ^bb0(%secret_inp: i16, %secret_cond: i1):
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

// CHECK-LABEL: @nested_secret_condition_with_secret_int
func.func @nested_secret_condition_with_secret_int(%inp: !secret.secret<i16>, %cond: !secret.secret<i1>) -> !secret.secret<i16> {
  // CHECK-NEXT: %[[C_2:.*]] = arith.constant 2 : i16
  // CHECK-NEXT: %[[C_10:.*]] = arith.constant 10 : i16
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[INP:.*]], %[[COND:.*]] : !secret.secret<[[T:.*]]>, !secret.secret<i1>) {
  // CHECK-NEXT:   ^[[bb0:.*]](%[[SCRT_INP:.*]]: [[T]], %[[SCRT_COND:.*]]: i1):
  // CHECK-NEXT:     %[[SUB:.*]] = arith.subi %[[SCRT_INP]], %[[C_10]] : [[T]]
  // CHECK-NEXT:     %[[CMP:.*]] = arith.cmpi slt, %[[SCRT_INP]], %[[C_10]] : [[T]]
  // CHECK-NEXT:      %[[SQR:.*]] = arith.muli %[[SCRT_INP]], %[[SCRT_INP]] : [[T]]
  // CHECK-NEXT:      %[[DBL:.*]] = arith.muli %[[SCRT_INP]], %[[C_2]] : [[T]]
  // CHECK-NEXT:     %[[SEL_CMP:.*]] = arith.select %[[CMP]], %[[SQR]], %[[DBL]] : [[T]]

  // CHECK-NEXT:     %[[ADD:.*]] = arith.addi %[[SEL_CMP]], %[[SUB]] : [[T]]
  // CHECK-NEXT:     %[[SEL_IF:.*]] = arith.select %[[SCRT_COND]], %[[ADD]], %[[SCRT_INP]] : [[T]]
  // CHECK-NEXT:     secret.yield %[[SEL_IF]] : [[T]]
  // CHECK-NEXT:  } -> !secret.secret<[[T]]>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<[[T]]>
  %c2_i16 = arith.constant 2 : i16
  %c10_i16 = arith.constant 10 : i16
  %0 = secret.generic ins(%inp, %cond : !secret.secret<i16>, !secret.secret<i1>) {
  ^bb0(%secret_inp: i16, %secret_cond: i1):
    %1 = scf.if %secret_cond -> (i16) {
      %2 = arith.subi %secret_inp, %c10_i16 : i16
      %3 = arith.cmpi slt, %secret_inp, %c10_i16 : i16
      %4 = scf.if %3 -> (i16) {
        // square the input if it is less than 10
        %6 = arith.muli %secret_inp, %secret_inp : i16
        scf.yield %6 : i16
      } else {
        // double the input if it is greater than or equal to 10
        %6 = arith.muli %c2_i16, %secret_inp : i16
        scf.yield %6 : i16
      }
      %5 = arith.addi %4, %2 : i16
      scf.yield %5 : i16
    } else {
      scf.yield %secret_inp : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
// CHECK-LABEL: @set_secretness_for_constant
func.func @set_secretness_for_constant(%arg0: !secret.secret<tensor<32xi16>>) -> !secret.secret<i16> {
  // CHECK-NEXT: %[[TEMP:.*]] = arith.constant 0 : [[C_T:.*]]
  // CHECK-NEXT: %[[RESULT:.*]] = secret.generic ins(%[[ARG0:.*]] : !secret.secret<tensor<[[T:.*]]>>) {
  // CHECK-NEXT:   ^[[bb0:.*]](%[[ARG1:.*]]: tensor<[[T]]>):
  // CHECK-NEXT:     %[[FOR:.*]] = affine.for %[[ARG2:.*]] = 0 to 32 iter_args(%[[ARG3:.*]] = %[[TEMP]]) -> (i16) {
  // CHECK-NEXT:       %[[EXTRACTED:.*]] = tensor.extract %[[ARG1]][%[[ARG2]]] : tensor<[[T]]>
  // CHECK-NEXT:       %[[COND:.*]] = arith.cmpi eq, %[[EXTRACTED]], %[[ARG3]] : [[C_T]]
  // CHECK-NEXT:       %[[ADD:.*]] = arith.addi %[[EXTRACTED]], %[[ARG3]] : [[C_T]]
  // CHECK-NEXT:       %[[SELECT:.*]] = arith.select %[[COND]], %[[ADD]], %[[EXTRACTED]] : [[C_T]]
  // CHECK-NEXT:       affine.yield %[[SELECT]] : [[C_T]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     secret.yield %[[FOR]] : [[C_T]]
  // CHECK-NEXT:   } -> !secret.secret<[[C_T]]>
  // CHECK-NEXT: return %[[RESULT]] : !secret.secret<[[C_T]]>
  %temp = arith.constant 0 : i16
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<32xi16>>) {
  ^bb0(%arg1: tensor<32xi16>):
      %result = affine.for %arg2 = 0 to 32 iter_args(%arg3 = %temp) -> (i16) {
          %extracted = tensor.extract %arg1[%arg2] : tensor<32xi16>
          %cond = arith.cmpi eq, %extracted, %arg3 : i16
          %1 = scf.if %cond -> (i16) {
              %2 = arith.addi %extracted, %arg3 : i16
              scf.yield %2 : i16
          } else {
              scf.yield %extracted : i16
          }
          affine.yield %1 : i16
      }
      secret.yield %result : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}


// CHECK-LABEL: @test_mixed_conditionals
// CHECK-SAME: ([[P:%.+]]: i1, [[S:%.+]]: !secret.secret<i1>, [[COND:%.+]]: i1)
func.func @test_mixed_conditionals(%p: i1, %s: !secret.secret<i1>, %cond: i1) -> !secret.secret<i1> {
  //CHECK: secret.generic ins([[S]]
  %0 = secret.generic ins(%s : !secret.secret<i1>) {
    //CHECK: ^[[bb0:.*]](%[[S_:.*]]: i1):
  ^bb0(%s_: i1):
    //CHECK-NEXT: [[IF:%.+]] = scf.if [[COND]]
    %1 = scf.if %cond -> (i1) {
      %ss = arith.addi %s_, %s_ : i1
      scf.yield %ss : i1
    } else {
      scf.yield %s_ : i1
    }
    //CHECK-NOT: scf.if
    //CHECK: [[ADD:%.+]] = arith.addi [[P]], [[P]]
    //CHECK: [[SEL:%.+]] = arith.select [[IF]], [[ADD]], [[P]]
      %2 = scf.if %1 -> (i1) {
      %pp = arith.addi %p, %p : i1
      scf.yield %pp : i1
    } else {
      scf.yield %p : i1
    }
    // CHECK-NOT: scf.if
    // CHECK: [[MUL:%.+]] = arith.muli [[P]], [[P]] : i1
    // CHECK: [[SEL2:%.+]] = arith.select [[SEL]], [[MUL]], [[P]]
    %3 = scf.if %2 -> (i1) {
      %pp = arith.muli %p, %p : i1
      scf.yield %pp : i1
    } else {
      scf.yield %p : i1
    }
    // CHECK: secret.yield [[SEL2]] : i1
    secret.yield %3 : i1
  } -> !secret.secret<i1>
  return %0 : !secret.secret<i1>
}

// CHECK-LABEL: @unknown_region
// CHECK-SAME: ([[P:%.+]]: i1, [[S:%.+]]: !secret.secret<i1>)
func.func @unknown_region(%p : i1, %s: !secret.secret<i1>) -> !secret.secret<i1> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: secret.generic ins([[S]]
  %0 = secret.generic ins(%s: !secret.secret<i1>) {
    //CHECK: ^[[bb0:.*]](%[[S_:.*]]: i1):
  ^bb0(%s_: i1):
    // CHECK: [[T:%.+]] = tensor.generate
    %t = tensor.generate %c1 {
      ^bb1(%i : index):
        %sp = arith.muli %p, %s_ : i1
        tensor.yield %sp : i1
    } : tensor<?xi1>
    // CHECK: [[T0:%.+]] = tensor.extract [[T]]
    %t0 = tensor.extract %t[%c0] : tensor<?xi1>
    // CHECK-NOT: scf.if
    // CHECK: [[ADD:%.+]] = arith.addi [[P]], [[P]]
    // CHECK: [[SEL:%.+]] = arith.select [[T0]], [[ADD]], [[P]]
    %if = scf.if %t0 -> (i1) {
      %pp = arith.addi %p, %p : i1
      scf.yield %pp : i1
    } else {
      scf.yield %p : i1
    }
    // CHECK: secret.yield [[SEL]]
    secret.yield %if : i1
  } -> !secret.secret<i1>
  return %0 : !secret.secret<i1>
}
