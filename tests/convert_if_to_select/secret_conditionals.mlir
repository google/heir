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
