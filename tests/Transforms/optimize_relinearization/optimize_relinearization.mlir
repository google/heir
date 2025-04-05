// RUN: heir-opt --mlir-print-local-scope --secretize --mlir-to-secret-arithmetic=ciphertext-degree=8 --optimize-relinearization %s | FileCheck %s

// CHECK: func.func @two_muls_followed_by_add
// CHECK: secret.generic
// CHECK: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-SAME: dimension = 3
// CHECK-NEXT: %[[RELINEARIZE0:.*]] = mgmt.relinearize
// CHECK-NEXT: secret.yield %[[RELINEARIZE0]]

func.func @two_muls_followed_by_add(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>, %arg2: tensor<8xi16>, %arg3: tensor<8xi16>) -> tensor<8xi16> {
  %0 = arith.muli %arg0, %arg1 : tensor<8xi16>
  %1 = mgmt.relinearize %0 : tensor<8xi16>

  %2 = arith.muli %arg2, %arg3 : tensor<8xi16>
  %3 = mgmt.relinearize %2 : tensor<8xi16>

  %z = arith.addi %1, %3 : tensor<8xi16>
  func.return %z : tensor<8xi16>
}

// CHECK: func.func @two_muls_followed_by_add_f16
// CHECK: secret.generic
// CHECK: arith.mulf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.addf
// CHECK-SAME: dimension = 3
// CHECK-NEXT: %[[RELINEARIZE0:.*]] = mgmt.relinearize
// CHECK-NEXT: secret.yield %[[RELINEARIZE0]]

func.func @two_muls_followed_by_add_f16(%arg0: tensor<8xf16>, %arg1: tensor<8xf16>, %arg2: tensor<8xf16>, %arg3: tensor<8xf16>) -> tensor<8xf16> {
  %0 = arith.mulf %arg0, %arg1 : tensor<8xf16>
  %1 = mgmt.relinearize %0 : tensor<8xf16>

  %2 = arith.mulf %arg2, %arg3 : tensor<8xf16>
  %3 = mgmt.relinearize %2 : tensor<8xf16>

  %z = arith.addf %1, %3 : tensor<8xf16>
  func.return %z : tensor<8xf16>
}

// CHECK: func.func @six_muls_with_add
// CHECK: secret.generic
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

func.func @six_muls_with_add(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>, %arg2: tensor<8xi16>, %arg3: tensor<8xi16>) -> tensor<8xi16> {
  %0 = arith.muli %arg0, %arg1 : tensor<8xi16>
  %1 = mgmt.relinearize %0  : tensor<8xi16>

  %2 = arith.muli %arg0, %arg2  : tensor<8xi16>
  %3 = mgmt.relinearize %2  : tensor<8xi16>

  %4 = arith.muli %arg0, %arg3  : tensor<8xi16>
  %5 = mgmt.relinearize %4  : tensor<8xi16>

  %6 = arith.muli %arg1, %arg2  : tensor<8xi16>
  %7 = mgmt.relinearize %6  : tensor<8xi16>

  %8 = arith.muli %arg1, %arg3  : tensor<8xi16>
  %9 = mgmt.relinearize %8  : tensor<8xi16>

  %10 = arith.muli %arg2, %arg3  : tensor<8xi16>
  %11 = mgmt.relinearize %10  : tensor<8xi16>

  %add1 = arith.addi %1, %3 : tensor<8xi16>
  %add2 = arith.addi %5, %7 : tensor<8xi16>
  %add3 = arith.addi %9, %11 : tensor<8xi16>
  %add4 = arith.addi %add1, %add2 : tensor<8xi16>
  %add5 = arith.addi %add3, %add4 : tensor<8xi16>
  func.return %add5 : tensor<8xi16>
}

// CHECK: func.func @six_muls_with_add_f16
// CHECK: secret.generic
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

func.func @six_muls_with_add_f16(%arg0: tensor<8xf16>, %arg1: tensor<8xf16>, %arg2: tensor<8xf16>, %arg3: tensor<8xf16>) -> tensor<8xf16> {
  %0 = arith.mulf %arg0, %arg1 : tensor<8xf16>
  %1 = mgmt.relinearize %0  : tensor<8xf16>

  %2 = arith.mulf %arg0, %arg2  : tensor<8xf16>
  %3 = mgmt.relinearize %2  : tensor<8xf16>

  %4 = arith.mulf %arg0, %arg3  : tensor<8xf16>
  %5 = mgmt.relinearize %4  : tensor<8xf16>

  %6 = arith.mulf %arg1, %arg2  : tensor<8xf16>
  %7 = mgmt.relinearize %6  : tensor<8xf16>

  %8 = arith.mulf %arg1, %arg3  : tensor<8xf16>
  %9 = mgmt.relinearize %8  : tensor<8xf16>

  %10 = arith.mulf %arg2, %arg3  : tensor<8xf16>
  %11 = mgmt.relinearize %10  : tensor<8xf16>

  %add1 = arith.addf %1, %3 : tensor<8xf16>
  %add2 = arith.addf %5, %7 : tensor<8xf16>
  %add3 = arith.addf %9, %11 : tensor<8xf16>
  %add4 = arith.addf %add1, %add2 : tensor<8xf16>
  %add5 = arith.addf %add3, %add4 : tensor<8xf16>
  func.return %add5 : tensor<8xf16>
}

// Test for a max key basis degree of 3, i.e., cannot do more than one repeated
// mul op before relinearizing.
// CHECK: func.func @repeated_mul
// CHECK: secret.generic
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

func.func @repeated_mul(%arg0: tensor<8xi16>) -> tensor<8xi16> {
  %0 = arith.muli %arg0, %arg0: tensor<8xi16>
  %1 = mgmt.relinearize %0  : tensor<8xi16>

  %2 = arith.muli %1, %1: tensor<8xi16>
  %3 = mgmt.relinearize %2  : tensor<8xi16>

  %4 = arith.muli %3, %3: tensor<8xi16>
  %5 = mgmt.relinearize %4  : tensor<8xi16>

  %6 = arith.muli %5, %5: tensor<8xi16>
  %7 = mgmt.relinearize %6  : tensor<8xi16>

  %8 = arith.muli %7, %7: tensor<8xi16>
  %9 = mgmt.relinearize %8  : tensor<8xi16>

  %z = arith.addi %9, %9 : tensor<8xi16>
  func.return %z : tensor<8xi16>
}

// Test that non mul/add ops work well with generic op handling in the analysis
// CHECK-LABEL: func.func @smoke_test
// CHECK-COUNT-5: arith.constant
// CHECK: secret.generic
// CHECK: arith.muli
// CHECK: arith.muli
// CHECK: arith.subi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: mgmt.relinearize
// CHECK-NEXT: secret.yield
func.func @smoke_test(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>) -> tensor<8xi16> {
  %cst = arith.constant dense<3> : tensor<8xi16>

  %0 = arith.muli %arg0, %arg0: tensor<8xi16>
  %1 = mgmt.relinearize %0  : tensor<8xi16>
  %2 = arith.muli %arg1, %arg1: tensor<8xi16>
  %3 = mgmt.relinearize %2  : tensor<8xi16>

  %c1 = arith.constant dense<1> : tensor<8xi16>
  %6 = arith.subi %3, %c1 : tensor<8xi16>
  %7 = arith.muli %1, %cst : tensor<8xi16>
  %8 = arith.addi %6, %7 : tensor<8xi16>
  func.return %8 : tensor<8xi16>
}

// CHECK: func.func @rotation_needs_linear_inputs
// CHECK: secret.generic
// CHECK: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: tensor_ext.rotate
// CHECK-NEXT: arith.addi
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: secret.yield
func.func @rotation_needs_linear_inputs(%arg0: tensor<8xi16>, %arg1: tensor<8xi16>) -> tensor<8xi16> {
  %0 = arith.muli %arg0, %arg0: tensor<8xi16>
  %1 = mgmt.relinearize %0  : tensor<8xi16>
  %2 = arith.muli %arg1, %arg1: tensor<8xi16>
  %3 = mgmt.relinearize %2  : tensor<8xi16>

  // Rotation requires degree 1 key basis input
  %c1 = arith.constant 1 : index
  %6 = tensor_ext.rotate %3, %c1 : tensor<8xi16>, index
  %7 = arith.addi %1, %6 : tensor<8xi16>
  func.return %7 : tensor<8xi16>
}

// CHECK: func.func @modreduce_needs_linear_inputs
// CHECK: secret.generic
// CHECK: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.subi
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK-NEXT: secret.yield
func.func @modreduce_needs_linear_inputs(%a: i64, %b: i64) -> (i64) {
    %0 = arith.muli %a, %a : i64
    %1 = arith.muli %b, %b : i64
    %2 = arith.subi %0, %1 : i64
    %ret = mgmt.modreduce %2 : i64
    func.return %ret : i64
}
