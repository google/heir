// RUN: heir-opt --optimize-relinearization %s | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=!mod_arith.int<161729713:i32>, polynomialModulus=#my_poly>
#params = #lwe.rlwe_params<dimension=2, ring=#ring>
#params1 = #lwe.rlwe_params<dimension=3, ring=#ring>

!pt = !lwe.rlwe_plaintext<encoding=#encoding, ring=#ring, underlying_type=i3>
!ct = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params, underlying_type=i3>
!ct1 = !lwe.rlwe_ciphertext<encoding=#encoding, rlwe_params=#params1, underlying_type=i3>

// CHECK-LABEL: func.func @two_muls_followed_by_add
// CHECK-SAME: %[[ARG0:[^:]*]]: !lwe.rlwe_ciphertext
// CHECK-SAME: %[[ARG1:[^:]*]]: !lwe.rlwe_ciphertext
// CHECK-SAME: %[[ARG2:[^:]*]]: !lwe.rlwe_ciphertext
// CHECK-SAME: %[[ARG3:[^:]*]]: !lwe.rlwe_ciphertext
// CHECK-NEXT: %[[MUL0:.*]] = bgv.mul %[[ARG0]], %[[ARG1]]
// CHECK-NEXT: %[[MUL1:.*]] = bgv.mul %[[ARG2]], %[[ARG3]]
// CHECK-NEXT: %[[ADD:.*]] = bgv.add %[[MUL0]], %[[MUL1]]
// CHECK-SAME: dimension = 3
// CHECK-NEXT: %[[RELINEARIZE0:.*]] = bgv.relinearize %[[ADD]]
// CHECK-NEXT: return %[[RELINEARIZE0]]

func.func @two_muls_followed_by_add(%arg0: !ct, %arg1: !ct, %arg2: !ct, %arg3: !ct) -> !ct {
  %0 = bgv.mul %arg0, %arg1  : (!ct, !ct) -> !ct1
  %1 = bgv.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %2 = bgv.mul %arg2, %arg3  : (!ct, !ct) -> !ct1
  %3 = bgv.relinearize %2  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %z = bgv.add %1, %3 : !ct
  func.return %z : !ct
}

// CHECK-LABEL: func.func @six_muls_with_add
// CHECK-SAME: %[[ARG0:[^:]*]]: !lwe.rlwe_ciphertext
// CHECK-SAME: %[[ARG1:[^:]*]]: !lwe.rlwe_ciphertext
// CHECK-SAME: %[[ARG2:[^:]*]]: !lwe.rlwe_ciphertext
// CHECK-SAME: %[[ARG3:[^:]*]]: !lwe.rlwe_ciphertext
// CHECK-NEXT: %[[MUL0:.*]] = bgv.mul %[[ARG0]], %[[ARG1]]
// CHECK-NEXT: %[[MUL1:.*]] = bgv.mul %[[ARG0]], %[[ARG2]]
// CHECK-NEXT: %[[MUL2:.*]] = bgv.mul %[[ARG0]], %[[ARG3]]
// CHECK-NEXT: %[[MUL3:.*]] = bgv.mul %[[ARG1]], %[[ARG2]]
// CHECK-NEXT: %[[MUL4:.*]] = bgv.mul %[[ARG1]], %[[ARG3]]
// CHECK-NEXT: %[[MUL5:.*]] = bgv.mul %[[ARG2]], %[[ARG3]]
// CHECK-NEXT: %[[ADD1:.*]] = bgv.add %[[MUL0]], %[[MUL1]]
// CHECK-NEXT: %[[ADD2:.*]] = bgv.add %[[MUL2]], %[[MUL3]]
// CHECK-NEXT: %[[ADD3:.*]] = bgv.add %[[MUL4]], %[[MUL5]]
// CHECK-NEXT: %[[ADD4:.*]] = bgv.add %[[ADD1]], %[[ADD2]]
// CHECK-NEXT: %[[ADD5:.*]] = bgv.add %[[ADD3]], %[[ADD4]]
// CHECK-NEXT: %[[RELINEARIZE0:.*]] = bgv.relinearize %[[ADD5]]
// CHECK-NEXT: return %[[RELINEARIZE0]]

func.func @six_muls_with_add(%arg0: !ct, %arg1: !ct, %arg2: !ct, %arg3: !ct) -> !ct {
  %0 = bgv.mul %arg0, %arg1  : (!ct, !ct) -> !ct1
  %1 = bgv.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %2 = bgv.mul %arg0, %arg2  : (!ct, !ct) -> !ct1
  %3 = bgv.relinearize %2  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %4 = bgv.mul %arg0, %arg3  : (!ct, !ct) -> !ct1
  %5 = bgv.relinearize %4  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %6 = bgv.mul %arg1, %arg2  : (!ct, !ct) -> !ct1
  %7 = bgv.relinearize %6  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %8 = bgv.mul %arg1, %arg3  : (!ct, !ct) -> !ct1
  %9 = bgv.relinearize %8  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %10 = bgv.mul %arg2, %arg3  : (!ct, !ct) -> !ct1
  %11 = bgv.relinearize %10  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %add1 = bgv.add %1, %3 : !ct
  %add2 = bgv.add %5, %7 : !ct
  %add3 = bgv.add %9, %11 : !ct
  %add4 = bgv.add %add1, %add2 : !ct
  %add5 = bgv.add %add3, %add4 : !ct
  func.return %add5 : !ct
}

// Test for a max key basis degree of 3, i.e., cannot do more than one repeated
// mul op before relinearizing.
// CHECK-LABEL: func.func @repeated_mul
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.relinearize
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.relinearize
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.relinearize
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.relinearize
// CHECK-NEXT: bgv.mul
// CHECK-DAG: bgv.add
// CHECK-DAG: bgv.relinearize
// CHECK-NEXT: return

func.func @repeated_mul(%arg0: !ct) -> !ct {
  %0 = bgv.mul %arg0, %arg0: (!ct, !ct) -> !ct1
  %1 = bgv.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %2 = bgv.mul %1, %1: (!ct, !ct) -> !ct1
  %3 = bgv.relinearize %2  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %4 = bgv.mul %3, %3: (!ct, !ct) -> !ct1
  %5 = bgv.relinearize %4  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %6 = bgv.mul %5, %5: (!ct, !ct) -> !ct1
  %7 = bgv.relinearize %6  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %8 = bgv.mul %7, %7: (!ct, !ct) -> !ct1
  %9 = bgv.relinearize %8  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %z = bgv.add %9, %9 : !ct
  func.return %z : !ct
}

// Test that non mul/add ops work well with generic op handling in the analysis
// CHECK-LABEL: func.func @smoke_test
// CHECK-NEXT: arith.constant
// CHECK-NEXT: lwe.rlwe_encode
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.negate
// CHECK-NEXT: bgv.mul_plain
// CHECK-NEXT: bgv.add
// CHECK-NEXT: bgv.relinearize
// CHECK-NEXT: return
func.func @smoke_test(%arg0: !ct, %arg1: !ct) -> !ct {
  %cst = arith.constant 3 : i3
  %plaintext = lwe.rlwe_encode %cst {encoding = #encoding, ring = #ring} : i3 -> !pt

  %0 = bgv.mul %arg0, %arg0: (!ct, !ct) -> !ct1
  %1 = bgv.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct
  %2 = bgv.mul %arg1, %arg1: (!ct, !ct) -> !ct1
  %3 = bgv.relinearize %2  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  %6 = bgv.negate %3 : !ct
  %7 = bgv.mul_plain %1, %plaintext : (!ct, !pt) -> !ct
  %8 = bgv.add %6, %7 : !ct
  func.return %8 : !ct
}

// CHECK-LABEL: func.func @rotation_needs_linear_inputs
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.relinearize
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.relinearize
// CHECK-NEXT: bgv.rotate
// CHECK-NEXT: bgv.add
// CHECK-NEXT: return
func.func @rotation_needs_linear_inputs(%arg0: !ct, %arg1: !ct) -> !ct {
  %0 = bgv.mul %arg0, %arg0: (!ct, !ct) -> !ct1
  %1 = bgv.relinearize %0  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct
  %2 = bgv.mul %arg1, %arg1: (!ct, !ct) -> !ct1
  %3 = bgv.relinearize %2  {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : !ct1 -> !ct

  // Rotation requires degree 1 key basis input
  %6 = bgv.rotate %3 { offset = 1 } : !ct
  %7 = bgv.add %1, %6 : !ct
  func.return %7 : !ct
}
