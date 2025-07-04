// RUN: heir-opt -mod-arith-to-arith --split-input-file %s | FileCheck %s --enable-var-scope

!Zp = !mod_arith.int<65537 : i32>
!Zpv = tensor<4x!Zp>

// CHECK: @test_lower_encapsulate
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_encapsulate(%lhs : i32) -> !Zp {
  // CHECK-NOT: mod_arith.encapsulate
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.encapsulate %lhs: i32 -> !Zp
  return %res : !Zp
}

// CHECK: @test_lower_encapsulate_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_encapsulate_vec(%lhs : tensor<4xi32>) -> !Zpv {
  // CHECK-NOT: mod_arith.encapsulate
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.encapsulate %lhs: tensor<4xi32> -> !Zpv
  return %res : !Zpv
}

// CHECK: @test_lower_extract
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_extract(%lhs : !Zp) -> i32 {
  // CHECK-NOT: mod_arith.extract
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.extract %lhs: !Zp -> i32
  return %res : i32
}

// CHECK: @test_lower_extract_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_extract_vec(%lhs : !Zpv) -> tensor<4xi32> {
  // CHECK-NOT: mod_arith.extract
  // CHECK: return %[[LHS]] : [[T]]
  %res = mod_arith.extract %lhs: !Zpv -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK: @test_lower_reduce
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_reduce(%lhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.reduce
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[REMS:.*]] = arith.remsi %[[LHS]], %[[CMOD]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[REMS]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.reduce %lhs: !Zp
  return %res : !Zp
}

// CHECK: @test_lower_reduce_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]]) -> [[T]] {
func.func @test_lower_reduce_vec(%lhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.reduce
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[REMS:.*]] = arith.remsi %[[LHS]], %[[CMOD]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[REMS]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.reduce %lhs: !Zpv
  return %res : !Zpv
}

// CHECK: @test_lower_add
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK: @test_lower_add_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_add_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.add
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.add %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK: @test_lower_sub
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK: @test_lower_sub_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_sub_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.sub
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[T]]
  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[T]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[SUB]], %[[CMOD]] : [[T]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[T]]
  // CHECK: return %[[REM]] : [[T]]
  %res = mod_arith.sub %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK: @test_lower_mul
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul(%lhs : !Zp, %rhs : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.mul
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mul %lhs, %rhs : !Zp
  return %res : !Zp
}

// CHECK: @test_lower_mul_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mul_vec(%lhs : !Zpv, %rhs : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.mul
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[MUL]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mul %lhs, %rhs : !Zpv
  return %res : !Zpv
}

// CHECK: @test_lower_mac
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]], %[[ACC:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mac(%lhs : !Zp, %rhs : !Zp, %acc : !Zp) -> !Zp {
  // CHECK-NOT: mod_arith.mac
  // CHECK: %[[CMOD:.*]] = arith.constant 65537 : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT2:.*]] = arith.extui %[[ACC]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[EXT2]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mac %lhs, %rhs, %acc : !Zp
  return %res : !Zp
}

// CHECK: @test_lower_mac_vec
// CHECK-SAME: (%[[LHS:.*]]: [[T:.*]], %[[RHS:.*]]: [[T]], %[[ACC:.*]]: [[T]]) -> [[T]] {
func.func @test_lower_mac_vec(%lhs : !Zpv, %rhs : !Zpv, %acc : !Zpv) -> !Zpv {
  // CHECK-NOT: mod_arith.mac
  // CHECK: %[[CMOD:.*]] = arith.constant dense<65537> : [[TEXT:.*]]
  // CHECK: %[[EXT0:.*]] = arith.extui %[[LHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT1:.*]] = arith.extui %[[RHS]] : [[T]] to [[TEXT]]
  // CHECK: %[[EXT2:.*]] = arith.extui %[[ACC]] : [[T]] to [[TEXT]]
  // CHECK: %[[MUL:.*]] = arith.muli %[[EXT0]], %[[EXT1]] : [[TEXT]]
  // CHECK: %[[ADD:.*]] = arith.addi %[[MUL]], %[[EXT2]] : [[TEXT]]
  // CHECK: %[[REM:.*]] = arith.remui %[[ADD]], %[[CMOD]] : [[TEXT]]
  // CHECK: %[[TRUNC:.*]] = arith.trunci %[[REM]] : [[TEXT]] to [[T]]
  // CHECK: return %[[TRUNC]] : [[T]]
  %res = mod_arith.mac %lhs, %rhs, %acc : !Zpv
  return %res : !Zpv
}

// -----

// CHECK: @test_lower_subifge
// CHECK-SAME: (%[[LHS:.*]]: [[TENSOR_TYPE:.*]], %[[RHS:.*]]: [[TENSOR_TYPE]]) -> [[TENSOR_TYPE]] {
func.func @test_lower_subifge(%lhs : tensor<4xi8>, %rhs : tensor<4xi8>) -> tensor<4xi8> {

  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[TENSOR_TYPE]]
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[TENSOR_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[LHS]] : tensor<4xi1>, [[TENSOR_TYPE]]
  %res = mod_arith.subifge %lhs, %rhs: tensor<4xi8>
  return %res : tensor<4xi8>
}

// -----

// CHECK: @test_lower_subifge_int
// CHECK-SAME: (%[[LHS:.*]]: [[INT_TYPE:.*]], %[[RHS:.*]]: [[INT_TYPE]]) -> [[INT_TYPE]] {
func.func @test_lower_subifge_int(%lhs : i8, %rhs : i8) -> i8 {

  // CHECK: %[[SUB:.*]] = arith.subi %[[LHS]], %[[RHS]] : [[INT_TYPE]]
  // CHECK: %[[CMP:.*]] = arith.cmpi uge, %[[LHS]], %[[RHS]] : [[INT_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[SUB]], %[[LHS]] : [[INT_TYPE]]
  %res = mod_arith.subifge %lhs, %rhs: i8
  return %res : i8
}

// -----

// CHECK: @test_lower_barrett_reduce

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
  %res = mod_arith.barrett_reduce %arg { modulus = 17 } : tensor<4xi10>

  // CHECK: return %[[RES]] : [[TENSOR_TYPE]]
  return %res : tensor<4xi10>
}

// -----

// CHECK: @test_lower_barrett_reduce_int
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
  %res = mod_arith.barrett_reduce %arg { modulus = 17 } : i10

  // CHECK: return %[[RES]] : [[INT_TYPE]]
  return %res : i10
}

// -----

!Zp = !mod_arith.int<3097973 : i26>
!Zp_larger_width = !mod_arith.int<65537 : i32>
!Zp_same_width = !mod_arith.int<33181787 : i26>
!Zp_smaller_width = !mod_arith.int<257 : i10>
!RNS = !rns.rns<!mod_arith.int<829 : i11>, !mod_arith.int<101 : i11>, !mod_arith.int<37 : i11>>

// CHECK: @test_lower_mod_switch_decompose
// CHECK-SAME: (%[[ARG:.*]]: [[INT_TYPE:.*]]) -> [[TENSOR_TYPE:.*]] {
func.func @test_lower_mod_switch_decompose(%arg : !Zp) -> !RNS {
  // CHECK: %[[CONST_829:.*]] = arith.constant 829 : [[INT_TYPE]]
  // CHECK: %[[REMUI_0:.*]] = arith.remui %[[ARG]], %[[CONST_829]] : [[INT_TYPE]]
  // CHECK: %[[TRUNC_0:.*]] = arith.trunci %[[REMUI_0]] : [[INT_TYPE]] to [[RNS_INT_TYPE:.*]]
  // CHECK: %[[CONST_101:.*]] = arith.constant 101 : [[INT_TYPE]]
  // CHECK: %[[REMUI_1:.*]] = arith.remui %[[ARG]], %[[CONST_101]] : [[INT_TYPE]]
  // CHECK: %[[TRUNC_1:.*]] = arith.trunci %[[REMUI_1]] : [[INT_TYPE]] to [[RNS_INT_TYPE]]
  // CHECK: %[[CONST_37:.*]] = arith.constant 37 : [[INT_TYPE]]
  // CHECK: %[[REMUI_2:.*]] = arith.remui %[[ARG]], %[[CONST_37]] : [[INT_TYPE]]
  // CHECK: %[[TRUNC_2:.*]] = arith.trunci %[[REMUI_2]] : [[INT_TYPE]] to [[RNS_INT_TYPE]]
  // CHECK: %[[RES:.*]] = tensor.from_elements %[[TRUNC_0]], %[[TRUNC_1]], %[[TRUNC_2]] : [[TENSOR_TYPE]]
  %res = mod_arith.mod_switch %arg : !Zp to !RNS

  // CHECK: return %[[RES]] : [[TENSOR_TYPE]]
  return %res : !RNS
}

// CHECK: @test_lower_mod_switch_interpolate
// CHECK-SAME: (%[[ARG:.*]]: [[TENSOR_TYPE:.*]]) -> [[INT_TYPE:.*]] {
func.func @test_lower_mod_switch_interpolate(%arg : !RNS) -> !Zp {
  // CHECK: %[[CONST_3097973:.*]] = arith.constant 3097973 : [[LARGE_INT_TYPE:.*]]
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : [[LARGE_INT_TYPE]]
  // CHECK: %[[CONST_1192103:.*]] = arith.constant 1192103 : [[LARGE_INT_TYPE]]
  // CHECK: %[[INDEX_0:.*]] = arith.constant 0 : [[INDEX_TYPE:.*]]
  // CHECK: %[[ARG_0:.*]] = tensor.extract %[[ARG]][%[[INDEX_0]]] : [[TENSOR_TYPE]]
  // CHECK: %[[EXTUI_0:.*]] = arith.extui %[[ARG_0]] : [[RNS_INT_TYPE:.*]] to [[LARGE_INT_TYPE]]
  // CHECK: %[[MULI_0:.*]] = arith.muli %[[EXTUI_0]], %[[CONST_1192103]] : [[LARGE_INT_TYPE]]
  // CHECK: %[[ADDI_0:.*]] = arith.addi %[[ZERO]], %[[MULI_0]] : [[LARGE_INT_TYPE]]
  // CHECK: %[[CONST_398749:.*]] = arith.constant 398749 : [[LARGE_INT_TYPE]]
  // CHECK: %[[INDEX_1:.*]] = arith.constant 1 : [[INDEX_TYPE]]
  // CHECK: %[[ARG_1:.*]] = tensor.extract %[[ARG]][%[[INDEX_1]]] : [[TENSOR_TYPE]]
  // CHECK: %[[EXTUI_1:.*]] = arith.extui %[[ARG_1]] : [[RNS_INT_TYPE]] to [[LARGE_INT_TYPE]]
  // CHECK: %[[MULI_1:.*]] = arith.muli %[[EXTUI_1]], %[[CONST_398749]] : [[LARGE_INT_TYPE]]
  // CHECK: %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[MULI_1]] : [[LARGE_INT_TYPE]]
  // CHECK: %[[CONST_1507122:.*]] = arith.constant 1507122 : [[LARGE_INT_TYPE]]
  // CHECK: %[[INDEX_2:.*]] = arith.constant 2 : [[INDEX_TYPE]]
  // CHECK: %[[ARG_2:.*]] = tensor.extract %[[ARG]][%[[INDEX_2]]] : [[TENSOR_TYPE]]
  // CHECK: %[[EXTUI_2:.*]] = arith.extui %[[ARG_2]] : [[RNS_INT_TYPE]] to [[LARGE_INT_TYPE]]
  // CHECK: %[[MULI_2:.*]] = arith.muli %[[EXTUI_2]], %[[CONST_1507122]] : [[LARGE_INT_TYPE]]
  // CHECK: %[[ADDI_2:.*]] = arith.addi %[[ADDI_1]], %[[MULI_2]] : [[LARGE_INT_TYPE]]
  // CHECK: %[[REMUI:.*]] = arith.remui %[[ADDI_2]], %[[CONST_3097973]] : [[LARGE_INT_TYPE]]
  // CHECK: %[[RES:.*]] = arith.trunci %[[REMUI]] : [[LARGE_INT_TYPE]] to [[INT_TYPE]]
  %res = mod_arith.mod_switch %arg : !RNS to !Zp

  // CHECK: return %[[RES]] : [[INT_TYPE]]
  return %res : !Zp
}

// CHECK: @lower_mod_switch_larger_width
// CHECK-SAME: (%[[ARG:.*]]: [[INT_TYPE:.*]]) -> [[LARGER_INT_TYPE:.*]] {
func.func @lower_mod_switch_larger_width(%arg : !Zp) -> !Zp_larger_width {
  // CHECK: %[[EXTUI:.*]] = arith.extui %[[ARG]] : [[INT_TYPE]] to [[LARGER_INT_TYPE]]
  // CHECK: %[[CONST_65537:.*]] = arith.constant 65537 : [[LARGER_INT_TYPE]]
  // CHECK: %[[RESULT_ALT_1:.*]] = arith.remui %[[EXTUI]], %[[CONST_65537]] : [[LARGER_INT_TYPE]]
  // CHECK: %[[CONST_1548986:.*]] = arith.constant 1548986 : [[LARGER_INT_TYPE]]
  // CHECK: %[[CMPI:.*]] = arith.cmpi ugt, %[[CONST_1548986]], %[[EXTUI]] : [[LARGER_INT_TYPE]]
  // CHECK: %[[CONST_47803:.*]] = arith.constant 47803 : [[LARGER_INT_TYPE]]
  // CHECK: %[[ADDI:.*]] = arith.addi %[[RESULT_ALT_1]], %[[CONST_47803]] : [[LARGER_INT_TYPE]]
  // CHECK: %[[CONST_65537_NEW:.*]] = arith.constant 65537 : [[LARGER_INT_TYPE]]
  // CHECK: %[[RESULT_ALT_2:.*]] = arith.remui %[[ADDI]], %[[CONST_65537_NEW]] : [[LARGER_INT_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMPI]], %[[RESULT_ALT_1]], %[[RESULT_ALT_2]] : [[LARGER_INT_TYPE]]
  %res = mod_arith.mod_switch %arg : !Zp to !Zp_larger_width

  // CHECK: return %[[RES]] : [[LARGER_INT_TYPE]]
  return %res : !Zp_larger_width
}

// CHECK: @lower_mod_switch_same_width
// CHECK-SAME: (%[[ARG:.*]]: [[INT_TYPE:.*]]) -> [[INT_TYPE:.*]] {
func.func @lower_mod_switch_same_width(%arg : !Zp) -> !Zp_same_width {
  // CHECK: %[[CONST_33181787:.*]] = arith.constant 33181787 : [[INT_TYPE]]
  // CHECK: %[[RESULT_ALT_1:.*]] = arith.remui %[[ARG]], %[[CONST_33181787]] : [[INT_TYPE]]
  // CHECK: %[[CONST_1548986:.*]] = arith.constant 1548986 : [[INT_TYPE]]
  // CHECK: %[[CMPI:.*]] = arith.cmpi ugt, %[[CONST_1548986]], %[[ARG]] : [[INT_TYPE]]
  // CHECK: %[[CONST_30083814:.*]] = arith.constant 30083814 : [[INT_TYPE]]
  // CHECK: %[[ADDI:.*]] = arith.addi %[[RESULT_ALT_1]], %[[CONST_30083814]] : [[INT_TYPE]]
  // CHECK: %[[CONST_33181787:.*]] = arith.constant 33181787 : [[INT_TYPE]]
  // CHECK: %[[RESULT_ALT_2:.*]] = arith.remui %[[ADDI]], %[[CONST_33181787]] : [[INT_TYPE]]
  // CHECK: %[[RES:.*]] = arith.select %[[CMPI]], %[[RESULT_ALT_1]], %[[RESULT_ALT_2]] : [[INT_TYPE]]
  %res = mod_arith.mod_switch %arg : !Zp to !Zp_same_width

  // CHECK: return %[[RES]] : [[INT_TYPE]]
  return %res : !Zp_same_width
}

// CHECK: @lower_mod_switch_smaller_width
// CHECK-SAME: (%[[ARG:.*]]: [[INT_TYPE:.*]]) -> [[SMALLER_INT_TYPE:.*]] {
func.func @lower_mod_switch_smaller_width(%arg : !Zp) -> !Zp_smaller_width {
  // CHECK: %[[CONST_257:.*]] = arith.constant 257 : [[INT_TYPE]]
  // CHECK: %[[RESULT_ALT_1:.*]] = arith.remui %[[ARG]], %[[CONST_257]] : [[INT_TYPE]]
  // CHECK: %[[CONST_1548986:.*]] = arith.constant 1548986 : [[INT_TYPE]]
  // CHECK: %[[CMPI:.*]] = arith.cmpi ugt, %[[CONST_1548986]], %[[ARG]] : [[INT_TYPE]]
  // CHECK: %[[CONST_162:.*]] = arith.constant 162 : [[INT_TYPE]]
  // CHECK: %[[ADDI:.*]] = arith.addi %[[RESULT_ALT_1]], %[[CONST_162]] : [[INT_TYPE]]
  // CHECK: %[[CONST_257_NEW:.*]] = arith.constant 257 : [[INT_TYPE]]
  // CHECK: %[[RESULT_ALT_2:.*]] = arith.remui %[[ADDI]], %[[CONST_257_NEW]] : [[INT_TYPE]]
  // CHECK: %[[SELECT:.*]] = arith.select %[[CMPI]], %[[RESULT_ALT_1]], %[[RESULT_ALT_2]] : [[INT_TYPE]]
  // CHECK: %[[RES:.*]] = arith.trunci %[[SELECT]] : [[INT_TYPE]] to [[SMALLER_INT_TYPE]]
  %res = mod_arith.mod_switch %arg : !Zp to !Zp_smaller_width

  // CHECK: return %[[RES]] : [[SMALLER_INT_TYPE]]
  return %res : !Zp_smaller_width
}
