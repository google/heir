// RUN: heir-opt -arith-ext-to-arith --split-input-file %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: @test_lower_simple_add
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_simple_add(%lhs : i8, %rhs : i8) -> i8 {
  // CHECK-NOT: arith_ext.add
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] : [[TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant 17 : [[TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TYPE]]
  // CHECK: return %[[REM]] : [[TYPE]]
  %res = arith_ext.add %lhs, %rhs {modulus = 17 }: i8
  return %res : i8
}

// CHECK-LABEL: @test_lower_simple_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_simple_add_vec(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {
  // CHECK-NOT: arith_ext.add
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] : [[TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant dense<17> : [[TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TYPE]]
  // CHECK: return %[[REM]] : [[TYPE]]
  %res = arith_ext.add %lhs, %rhs {modulus = 17}: tensor<4xi8>
  return %res : tensor<4xi8>
}

// CHECK-LABEL: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_add(%lhs : i8, %rhs : i8) -> i8 {
  // CHECK-NOT: arith_ext.add
  // CHECK: %[[CMOD:.*]] = arith.constant 217 : [[INTERMEDIATE_TYPE:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[EXT0]], %[[EXT1]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[INTERMEDIATE_TYPE]] to [[TYPE]]
  // CHECK: return %[[TRUNC]] : [[TYPE]]
  %res = arith_ext.add %lhs, %rhs {modulus = 217 }: i8
  return %res : i8
}

// CHECK-LABEL: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_add_vec(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {
  // CHECK-NOT: arith_ext.add
  // CHECK: %[[CMOD:.*]] = arith.constant dense<217> : [[INTERMEDIATE_TYPE:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[EXT0]], %[[EXT1]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[INTERMEDIATE_TYPE]] to [[TYPE]]
  // CHECK: return %[[TRUNC]] : [[TYPE]]
  %res = arith_ext.add %lhs, %rhs {modulus = 217 }: tensor<4xi8>
  return %res : tensor<4xi8>
}

// CHECK-LABEL: @test_lower_simple_sub
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_simple_sub(%lhs : i8, %rhs : i8) -> i8 {
  // CHECK-NOT: arith_ext.sub
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant 17 : [[TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[SUB]], %[[CMOD]] : [[TYPE]]
  // CHECK: return %[[REM]] : [[TYPE]]
  %res = arith_ext.sub %lhs, %rhs {modulus = 17}: i8
  return %res : i8
}

// CHECK-LABEL: @test_lower_simple_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_simple_sub_vec(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {
  // CHECK-NOT: arith_ext.sub
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant dense<17> : [[TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[SUB]], %[[CMOD]] : [[TYPE]]
  // CHECK: return %[[REM]] : [[TYPE]]
  %res = arith_ext.sub %lhs, %rhs {modulus = 17}: tensor<4xi8>
  return %res : tensor<4xi8>
}

// CHECK-LABEL: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_sub(%lhs : i8, %rhs : i8) -> i8 {
  // CHECK-NOT: arith_ext.sub
  // CHECK: %[[CMOD:.*]] = arith.constant 217 : [[INTERMEDIATE_TYPE:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[EXT0]], %[[EXT1]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[SUB]], %[[CMOD]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[INTERMEDIATE_TYPE]] to [[TYPE]]
  // CHECK: return %[[TRUNC]] : [[TYPE]]
  %res = arith_ext.sub %lhs, %rhs {modulus = 217 }: i8
  return %res : i8
}

// CHECK-LABEL: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_sub_vec(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {
  // CHECK-NOT: arith_ext.sub
  // CHECK: %[[CMOD:.*]] = arith.constant dense<217> : [[INTERMEDIATE_TYPE:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[EXT0]], %[[EXT1]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[SUB]], %[[CMOD]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[INTERMEDIATE_TYPE]] to [[TYPE]]
  // CHECK: return %[[TRUNC]] : [[TYPE]]
  %res = arith_ext.sub %lhs, %rhs {modulus = 217 }: tensor<4xi8>
  return %res : tensor<4xi8>
}

// CHECK-LABEL: @test_lower_simple_mul
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_simple_mul(%lhs : i16, %rhs : i16) -> i16 {
  // CHECK-NOT: arith_ext.mul
  // CHECK: %[[MUL:.*]] = arith.muli %[[LHS]], %[[RHS]] : [[TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant 17 : [[TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[TYPE]]
  // CHECK: return %[[REM]] : [[TYPE]]
  %res = arith_ext.mul %lhs, %rhs {modulus = 17}: i16
  return %res : i16
}

// CHECK-LABEL: @test_lower_simple_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_simple_mul_vec(%lhs : tensor<4xi16>, %rhs : tensor<4xi16>) -> tensor<4xi16> {
  // CHECK-NOT: arith_ext.mul
  // CHECK: %[[MUL:.*]] = arith.muli %[[LHS]], %[[RHS]] : [[TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant dense<17> : [[TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[TYPE]]
  // CHECK: return %[[REM]] : [[TYPE]]
  %res = arith_ext.mul %lhs, %rhs {modulus = 17}: tensor<4xi16>
  return %res : tensor<4xi16>
}

// CHECK-LABEL: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_mul(%lhs : i8, %rhs : i8) -> i8 {
  // CHECK-NOT: arith_ext.mul
  // CHECK: %[[CMOD:.*]] = arith.constant 217 : [[INTERMEDIATE_TYPE:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[INTERMEDIATE_TYPE]] to [[TYPE]]
  // CHECK: return %[[TRUNC]] : [[TYPE]]
  %res = arith_ext.mul %lhs, %rhs {modulus = 217 }: i8
  return %res : i8
}

// CHECK-LABEL: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_mul_vec(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {
  // CHECK-NOT: arith_ext.mul
  // CHECK: %[[CMOD:.*]] = arith.constant dense<217> : [[INTERMEDIATE_TYPE:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[INTERMEDIATE_TYPE]] to [[TYPE]]
  // CHECK: return %[[TRUNC]] : [[TYPE]]
  %res = arith_ext.mul %lhs, %rhs {modulus = 217 }: tensor<4xi8>
  return %res : tensor<4xi8>
}

// CHECK-LABEL: @test_lower_simple_mac
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]], %[[ACC:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_simple_mac(%lhs : tensor<4xi16>, %rhs : tensor<4xi16>, %acc : tensor<4xi16>) -> tensor<4xi16> {
  // CHECK-NOT: arith_ext.mac
  // CHECK: %[[MUL:.*]] = arith.muli %[[LHS]], %[[RHS]] : [[TYPE]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[ACC]] : [[TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant dense<17> : [[TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TYPE]]
  // CHECK: return %[[REM]] : [[TYPE]]
  %res = arith_ext.mac %lhs, %rhs, %acc {modulus = 17}: tensor<4xi16>
  return %res : tensor<4xi16>
}

// CHECK-LABEL: @test_lower_simple_mac_vec
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]], %[[ACC:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_simple_mac_vec(%lhs : i16, %rhs : i16, %acc : i16) -> i16 {
  // CHECK-NOT: arith_ext.mac
  // CHECK: %[[MUL:.*]] = arith.muli %[[LHS]], %[[RHS]] : [[TYPE]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[ACC]] : [[TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant 17 : [[TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TYPE]]
  // CHECK: return %[[REM]] : [[TYPE]]
  %res = arith_ext.mac %lhs, %rhs, %acc{modulus = 17}: i16
  return %res : i16
}

// CHECK-LABEL: @test_lower_mac
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]], %[[ACC:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_mac(%lhs : i8, %rhs : i8, %acc : i8) -> i8 {
  // CHECK-NOT: arith_ext.mac
  // CHECK: %[[CMOD:.*]] = arith.constant 217 : [[INTERMEDIATE_TYPE:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT2:.*]] = arith.extui %[[ACC]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[EXT2]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[INTERMEDIATE_TYPE]] to [[TYPE]]
  // CHECK: return %[[TRUNC]] : [[TYPE]]
  %res = arith_ext.mac %lhs, %rhs, %acc {modulus = 217 }: i8
  return %res : i8
}

// CHECK-LABEL: @test_lower_mac_vec
// CHECK-SAME: (%[[LHS:.*]]: [[TYPE:.*]], %[[RHS:.*]]: [[TYPE]], %[[ACC:.*]]: [[TYPE]]) -> [[TYPE]] {
func.func @test_lower_mac_vec(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>, %acc : tensor<4xi8>) -> tensor<4xi8> {
  // CHECK-NOT: arith_ext.mac
  // CHECK: %[[CMOD:.*]] = arith.constant dense<217> : [[INTERMEDIATE_TYPE:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[EXT2:.*]] = arith.extui %[[ACC]] : [[TYPE]] to [[INTERMEDIATE_TYPE]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[EXT2]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[INTERMEDIATE_TYPE]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[INTERMEDIATE_TYPE]] to [[TYPE]]
  // CHECK: return %[[TRUNC]] : [[TYPE]]
  %res = arith_ext.mac %lhs, %rhs, %acc {modulus = 217 }: tensor<4xi8>
  return %res : tensor<4xi8>
}


// -----

// CHECK-LABEL: @test_lower_subifge
// CHECK-SAME: (%[[LHS:.*]]: [[TENSOR_TYPE:.*]], %[[RHS:.*]]: [[TENSOR_TYPE]]) -> [[TENSOR_TYPE]] {
func.func @test_lower_subifge(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {

  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[TENSOR_TYPE]]
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[TENSOR_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[LHS]] : tensor<4xi1>, [[TENSOR_TYPE]]
  %res = arith_ext.subifge %lhs, %rhs: tensor<4xi8>
  return %res : tensor<4xi8>
}

// -----

// CHECK-LABEL: @test_lower_subifge_int
// CHECK-SAME: (%[[LHS:.*]]: [[INT_TYPE:.*]], %[[RHS:.*]]: [[INT_TYPE]]) -> [[INT_TYPE]] {
func.func @test_lower_subifge_int(%lhs : i8, %rhs : i8) -> i8 {

  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[INT_TYPE]]
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[INT_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[LHS]] : [[INT_TYPE]]
  %res = arith_ext.subifge %lhs, %rhs: i8
  return %res : i8
}

// -----

// CHECK-LABEL: @test_lower_barrett_reduce

// CHECK-SAME: (%[[ARG:.*]]: [[TENSOR_TYPE:.*]]) -> [[TENSOR_TYPE]] {
func.func @test_lower_barrett_reduce(%arg : tensor<4xi10>) -> tensor<4xi10> {

  // CHECK: %[[RATIO:.*]] = arith.constant dense<60> : [[INTER_TYPE:.*]]
  // CHECK: %[[BITWIDTH:.*]] = arith.constant dense<10> : [[INTER_TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant dense<17> : [[INTER_TYPE]]

  // CHECK: %[[EXT:.*]] = arith.extui %[[ARG]] : [[TENSOR_TYPE]] to [[INTER_TYPE]]
  // CHECK: %[[MULRATIO:.*]] = arith.muli %[[EXT]], %[[RATIO]] : [[INTER_TYPE]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui %[[MULRATIO]], %[[BITWIDTH]] : [[INTER_TYPE]]
  // CHECK: %[[MULCMOD:.*]] = arith.muli %[[SHIFTED]], %[[CMOD]] : [[INTER_TYPE]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[EXT]], %[[MULCMOD]] : [[INTER_TYPE]]
  // CHECK: %[[RES:.*]] = arith.trunci %[[SUB]] : [[INTER_TYPE]] to [[TENSOR_TYPE]]
  %res = arith_ext.barrett_reduce %arg { modulus = 17 } : tensor<4xi10>

  // CHECK: return %[[RES]] : [[TENSOR_TYPE]]
  return %res : tensor<4xi10>
}

// -----

// CHECK-LABEL: @test_lower_barrett_reduce_int
// CHECK-SAME: (%[[ARG:.*]]: [[INT_TYPE:.*]]) -> [[INT_TYPE]] {
func.func @test_lower_barrett_reduce_int(%arg : i10) -> i10 {

  // CHECK: %[[RATIO:.*]] = arith.constant 60 : [[INTER_TYPE:.*]]
  // CHECK: %[[BITWIDTH:.*]] = arith.constant 10 : [[INTER_TYPE]]
  // CHECK: %[[CMOD:.*]] = arith.constant 17 : [[INTER_TYPE]]

  // CHECK: %[[EXT:.*]] = arith.extui %[[ARG]] : [[INT_TYPE]] to [[INTER_TYPE]]
  // CHECK: %[[MULRATIO:.*]] = arith.muli %[[EXT]], %[[RATIO]] : [[INTER_TYPE]]
  // CHECK: %[[SHIFTED:.*]] = arith.shrui %[[MULRATIO]], %[[BITWIDTH]] : [[INTER_TYPE]]
  // CHECK: %[[MULCMOD:.*]] = arith.muli %[[SHIFTED]], %[[CMOD]] : [[INTER_TYPE]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[EXT]], %[[MULCMOD]] : [[INTER_TYPE]]
  // CHECK: %[[RES:.*]] = arith.trunci %[[SUB]] : [[INTER_TYPE]] to [[INT_TYPE]]
  %res = arith_ext.barrett_reduce %arg { modulus = 17 } : i10

  // CHECK: return %[[RES]] : [[INT_TYPE]]
  return %res : i10
}

// -----
