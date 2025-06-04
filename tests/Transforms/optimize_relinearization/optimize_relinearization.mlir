// RUN: heir-opt --optimize-relinearization %s | FileCheck %s

// CHECK: func.func @two_muls_followed_by_add
// CHECK: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-SAME: dimension = 3
// CHECK-NEXT: %[[RELINEARIZE0:.*]] = mgmt.relinearize
// CHECK-NEXT: secret.yield %[[RELINEARIZE0]]

func.func @two_muls_followed_by_add(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>, %arg2: !secret.secret<tensor<8xi16>>, %arg3: !secret.secret<tensor<8xi16>>) -> !secret.secret<tensor<8xi16>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>, %arg2: !secret.secret<tensor<8xi16>>, %arg3: !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>, %input2: tensor<8xi16>, %input3: tensor<8xi16>):
    %1 = arith.muli %input0, %input1 : tensor<8xi16>
    %2 = mgmt.relinearize %1 : tensor<8xi16>
    %3 = arith.muli %input2, %input3 : tensor<8xi16>
    %4 = mgmt.relinearize %3 : tensor<8xi16>
    %5 = arith.addi %2, %4 : tensor<8xi16>
    secret.yield %5 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}

// CHECK: func.func @two_muls_followed_by_add_f16
// CHECK: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.addf
// CHECK-SAME: dimension = 3
// CHECK-NEXT: %[[RELINEARIZE0:.*]] = mgmt.relinearize
// CHECK-NEXT: secret.yield %[[RELINEARIZE0]]

func.func @two_muls_followed_by_add_f16(%arg0: !secret.secret<tensor<8xf16>>, %arg1: !secret.secret<tensor<8xf16>>, %arg2: !secret.secret<tensor<8xf16>>, %arg3: !secret.secret<tensor<8xf16>>) -> (!secret.secret<tensor<8xf16>>) {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xf16>>, %arg1: !secret.secret<tensor<8xf16>>, %arg2: !secret.secret<tensor<8xf16>>, %arg3: !secret.secret<tensor<8xf16>>) {
  ^body(%input0: tensor<8xf16>, %input1: tensor<8xf16>, %input2: tensor<8xf16>, %input3: tensor<8xf16>):
    %1 = arith.mulf %input0, %input1 : tensor<8xf16>
    %2 = mgmt.relinearize %1 : tensor<8xf16>
    %3 = arith.mulf %input2, %input3 : tensor<8xf16>
    %4 = mgmt.relinearize %3 : tensor<8xf16>
    %5 = arith.addf %2, %4 : tensor<8xf16>
    secret.yield %5 : tensor<8xf16>
  } -> !secret.secret<tensor<8xf16>>
  return %0 : !secret.secret<tensor<8xf16>>
}

// CHECK: func.func @six_muls_with_add
// CHECK: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.addi
// CHECK-SAME: dimension = 3
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: secret.yield

func.func @six_muls_with_add(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>, %arg2: !secret.secret<tensor<8xi16>>, %arg3: !secret.secret<tensor<8xi16>>) -> (!secret.secret<tensor<8xi16>>) {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>, %arg2: !secret.secret<tensor<8xi16>>, %arg3: !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>, %input2: tensor<8xi16>, %input3: tensor<8xi16>):
    %1 = arith.muli %input0, %input1 : tensor<8xi16>
    %2 = mgmt.relinearize %1 : tensor<8xi16>
    %3 = arith.muli %input0, %input2 : tensor<8xi16>
    %4 = mgmt.relinearize %3 : tensor<8xi16>
    %5 = arith.muli %input0, %input3 : tensor<8xi16>
    %6 = mgmt.relinearize %5 : tensor<8xi16>
    %7 = arith.muli %input1, %input2 : tensor<8xi16>
    %8 = mgmt.relinearize %7 : tensor<8xi16>
    %9 = arith.muli %input1, %input3 : tensor<8xi16>
    %10 = mgmt.relinearize %9 : tensor<8xi16>
    %11 = arith.muli %input2, %input3 : tensor<8xi16>
    %12 = mgmt.relinearize %11 : tensor<8xi16>
    %13 = arith.addi %10, %12 : tensor<8xi16>
    %14 = arith.addi %13, %2 : tensor<8xi16>
    %15 = arith.addi %4, %6 : tensor<8xi16>
    %16 = arith.addi %15, %8 : tensor<8xi16>
    %17 = arith.addi %14, %16 : tensor<8xi16>
    secret.yield %17 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}

// CHECK: func.func @six_muls_with_add_f16
// CHECK: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.addf
// CHECK-SAME: dimension = 3
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: secret.yield

func.func @six_muls_with_add_f16(%arg0: !secret.secret<tensor<8xf16>>, %arg1: !secret.secret<tensor<8xf16>>, %arg2: !secret.secret<tensor<8xf16>>, %arg3: !secret.secret<tensor<8xf16>>) -> (!secret.secret<tensor<8xf16>>) {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xf16>>, %arg1: !secret.secret<tensor<8xf16>>, %arg2: !secret.secret<tensor<8xf16>>, %arg3: !secret.secret<tensor<8xf16>>) {
  ^body(%input0: tensor<8xf16>, %input1: tensor<8xf16>, %input2: tensor<8xf16>, %input3: tensor<8xf16>):
    %1 = arith.mulf %input0, %input1 : tensor<8xf16>
    %2 = mgmt.relinearize %1 : tensor<8xf16>
    %3 = arith.mulf %input0, %input2 : tensor<8xf16>
    %4 = mgmt.relinearize %3 : tensor<8xf16>
    %5 = arith.mulf %input0, %input3 : tensor<8xf16>
    %6 = mgmt.relinearize %5 : tensor<8xf16>
    %7 = arith.mulf %input1, %input2 : tensor<8xf16>
    %8 = mgmt.relinearize %7 : tensor<8xf16>
    %9 = arith.mulf %input1, %input3 : tensor<8xf16>
    %10 = mgmt.relinearize %9 : tensor<8xf16>
    %11 = arith.mulf %input2, %input3 : tensor<8xf16>
    %12 = mgmt.relinearize %11 : tensor<8xf16>
    %13 = arith.addf %10, %12 : tensor<8xf16>
    %14 = arith.addf %13, %2 : tensor<8xf16>
    %15 = arith.addf %4, %6 : tensor<8xf16>
    %16 = arith.addf %15, %8 : tensor<8xf16>
    %17 = arith.addf %14, %16 : tensor<8xf16>
    secret.yield %17 : tensor<8xf16>
  } -> !secret.secret<tensor<8xf16>>
  return %0 : !secret.secret<tensor<8xf16>>
}

// Test for a max key basis degree of 3, i.e., cannot do more than one repeated
// mul op before relinearizing.
// CHECK: func.func @repeated_mul
// CHECK: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: arith.muli
// CHECK-DAG: arith.addi
// CHECK-DAG: mgmt.relinearize
// CHECK-NEXT: secret.yield

func.func @repeated_mul(%arg0: !secret.secret<tensor<8xi16>>) -> (!secret.secret<tensor<8xi16>>) {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>):
    %1 = arith.muli %input0, %input0 : tensor<8xi16>
    %2 = mgmt.relinearize %1 : tensor<8xi16>
    %3 = arith.muli %2, %2 : tensor<8xi16>
    %4 = mgmt.relinearize %3 : tensor<8xi16>
    %5 = arith.muli %4, %4 : tensor<8xi16>
    %6 = mgmt.relinearize %5 : tensor<8xi16>
    %7 = arith.muli %6, %6 : tensor<8xi16>
    %8 = mgmt.relinearize %7 : tensor<8xi16>
    %9 = arith.muli %8, %8 : tensor<8xi16>
    %10 = mgmt.relinearize %9 : tensor<8xi16>
    %11 = arith.addi %10, %10 : tensor<8xi16>
    secret.yield %11 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}

// Test that non mul/add ops work well with generic op handling in the analysis
// CHECK: func.func @smoke_test
// CHECK-COUNT-5: arith.constant
// CHECK: secret.generic
// CHECK: arith.muli
// CHECK: arith.muli
// CHECK: arith.subi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: mgmt.relinearize
// CHECK-NEXT: secret.yield
func.func @smoke_test(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>) -> (!secret.secret<tensor<8xi16>>) {
  %c3_i16 = arith.constant 3 : i16
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1_i16 = arith.constant 1 : i16
  %cst = arith.constant dense<0> : tensor<8xi16>
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>):
    %1 = arith.muli %input0, %input0 : tensor<8xi16>
    %2 = mgmt.relinearize %1 : tensor<8xi16>
    %3 = arith.muli %input1, %input1 : tensor<8xi16>
    %4 = mgmt.relinearize %3 : tensor<8xi16>
    %5 = affine.for %arg2 = 0 to 8 iter_args(%arg3 = %cst) -> (tensor<8xi16>) {
      %10 = arith.remsi %arg2, %c8 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c8 : index
      %13 = arith.select %11, %12, %10 : index
      %inserted = tensor.insert %c1_i16 into %arg3[%13] : tensor<8xi16>
      affine.yield %inserted : tensor<8xi16>
    }
    %6 = arith.subi %4, %5 : tensor<8xi16>
    %7 = affine.for %arg2 = 0 to 8 iter_args(%arg3 = %cst) -> (tensor<8xi16>) {
      %10 = arith.remsi %arg2, %c8 : index
      %11 = arith.cmpi slt, %10, %c0 : index
      %12 = arith.addi %10, %c8 : index
      %13 = arith.select %11, %12, %10 : index
      %inserted = tensor.insert %c3_i16 into %arg3[%13] : tensor<8xi16>
      affine.yield %inserted : tensor<8xi16>
    }
    %8 = arith.muli %2, %7 : tensor<8xi16>
    %9 = arith.addi %6, %8 : tensor<8xi16>
    secret.yield %9 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}

// CHECK: func.func @rotation_needs_linear_inputs
// CHECK: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: tensor_ext.rotate
// CHECK-NEXT: arith.addi
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: secret.yield
func.func @rotation_needs_linear_inputs(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>) -> (!secret.secret<tensor<8xi16>>) {
  %c1 = arith.constant 1 : index
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>>, %arg1: !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>):
    %1 = arith.muli %input0, %input0 : tensor<8xi16>
    %2 = mgmt.relinearize %1 : tensor<8xi16>
    %3 = arith.muli %input1, %input1 : tensor<8xi16>
    %4 = mgmt.relinearize %3 : tensor<8xi16>
    // Rotation requires linear key basis for its input
    %5 = tensor_ext.rotate %4, %c1 : tensor<8xi16>, index
    %6 = arith.addi %2, %5 : tensor<8xi16>
    secret.yield %6 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}

// CHECK: func.func @modreduce_needs_linear_inputs
// CHECK: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.subi
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK-NEXT: secret.yield
func.func @modreduce_needs_linear_inputs(%arg0: !secret.secret<tensor<8xi64>>, %arg1: !secret.secret<tensor<8xi64>>) -> (!secret.secret<tensor<8xi64>>) {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi64>>, %arg1: !secret.secret<tensor<8xi64>>) {
  ^body(%input0: tensor<8xi64>, %input1: tensor<8xi64>):
    %1 = arith.muli %input0, %input0 : tensor<8xi64>
    %2 = arith.muli %input1, %input1 : tensor<8xi64>
    %3 = arith.subi %1, %2 : tensor<8xi64>
    %4 = mgmt.modreduce %3 : tensor<8xi64>
    secret.yield %4 : tensor<8xi64>
  } -> !secret.secret<tensor<8xi64>>
  return %0 : !secret.secret<tensor<8xi64>>
}
