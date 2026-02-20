// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
#ring_1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
!poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=coeff>
!ntt_poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=eval>

#ring_2 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1>, polynomialModulus = <1 + x**1024>>
!poly_ty_2 = !polynomial.polynomial<ring=#ring_2, form=coeff>
!ntt_poly_ty_2 = !polynomial.polynomial<ring=#ring_2, form=eval>

#ring_3 = #polynomial.ring<coefficientType = !Zq0, polynomialModulus = <1 + x**8>>
!poly_ty_3 = !polynomial.polynomial<ring=#ring_3, form=coeff>
!ntt_poly_ty_3 = !polynomial.polynomial<ring=#ring_3, form=eval>

#ring_4 = #polynomial.ring<coefficientType = !Zq1, polynomialModulus = <1 + x**8>>
!poly_ty_4 = !polynomial.polynomial<ring=#ring_4, form=coeff>
!ntt_poly_ty_4 = !polynomial.polynomial<ring=#ring_4, form=eval>

module {
  // Covers: baseline MulOp rewrite from coeff-form input to eval-form signature/result.
  // CHECK: func.func @test_ntt_insertion1([[x:%.+]]: [[T1:![^ ]+]]) -> [[T1]] {
  func.func @test_ntt_insertion1(%x: !poly_ty_1) -> !poly_ty_1 {
    // CHECK-DAG: [[xsq:%.+]] = polynomial.mul [[x]], [[x]] : [[T1]]
    %xsq = polynomial.mul %x, %x : !poly_ty_1
    // CHECK-DAG: return [[xsq]] : [[T1]]
    return %xsq: !poly_ty_1
  }

  // Covers: eval->coeff path with INTT + ConvertBasis to satisfy narrower output basis.
  // CHECK: func.func @test_ntt_insertion2([[x2:%.+]]: [[T2E:![^ ]+]]) -> [[T1C:![^ ]+]] {
  func.func @test_ntt_insertion2(%x: !poly_ty_2) -> !poly_ty_1 {
    // CHECK-DAG: [[xsq2:%.+]] = polynomial.mul [[x2]], [[x2]] : [[T2E]]
    %xsq = polynomial.mul %x, %x : !poly_ty_2
    // CHECK-DAG: [[coeff2:%.+]] = polynomial.intt [[xsq2]] : [[T2E]]
    // CHECK-DAG: [[out2:%.+]] = polynomial.convert_basis [[coeff2]] {targetBasis = {{.*}}} : [[T2C:![^ ]+]]
    %y = polynomial.convert_basis %xsq {targetBasis = !rns.rns<!Zq0>} : !poly_ty_2
    // CHECK-DAG: return [[out2]] : [[T1C]]
    return %y: !poly_ty_1
  }

  // Covers: flexible Add/Sub mode + forced eval by downstream MulOp in same function.
  // CHECK: func.func @test_ntt_insertion3([[x3:%.+]]: [[T3IN:![^ ]+]]) -> [[T3OUT:![^ ]+]] {
  func.func @test_ntt_insertion3(%x: !poly_ty_1) -> !poly_ty_1 {
    // CHECK-DAG: [[x3e:%.+]] = polynomial.ntt [[x3]] : [[T3IN]]
    // CHECK-DAG: [[a3:%.+]] = polynomial.add [[x3e]], [[x3e]] : [[T3OUT]]
    %a = polynomial.add %x, %x : !poly_ty_1
    // CHECK-DAG: [[b3:%.+]] = polynomial.sub [[a3]], [[x3e]] : [[T3OUT]]
    %b = polynomial.sub %a, %x : !poly_ty_1
    // CHECK-DAG: [[c3:%.+]] = polynomial.mul [[b3]], [[b3]] : [[T3OUT]]
    %c = polynomial.mul %b, %b : !poly_ty_1
    // CHECK-DAG: return [[c3]] : [[T3OUT]]
    return %c : !poly_ty_1
  }

  // Covers: another basis-conversion path to exercise solver consistency across functions.
  // CHECK: func.func @test_ntt_insertion4([[x4:%.+]]: [[T4E:![^ ]+]]) -> [[T4C:![^ ]+]] {
  func.func @test_ntt_insertion4(%x: !poly_ty_2) -> !poly_ty_1 {
    // CHECK-DAG: [[xsq4:%.+]] = polynomial.mul [[x4]], [[x4]] : [[T4E]]
    %xsq = polynomial.mul %x, %x : !poly_ty_2
    // CHECK-DAG: [[coeff4:%.+]] = polynomial.intt [[xsq4]] : [[T4E]]
    // CHECK-DAG: [[out4:%.+]] = polynomial.convert_basis [[coeff4]] {targetBasis = {{.*}}} : [[T4CIN:![^ ]+]]
    %out = polynomial.convert_basis %xsq {targetBasis = !rns.rns<!Zq0>} : !poly_ty_2
    // CHECK-DAG: return [[out4]] : [[T4C]]
    return %out : !poly_ty_1
  }

  // Covers: flexible-op-only pipeline (Add/Sub) without mandatory coeff/eval-only consumers.
  // CHECK: func.func @test_ntt_insertion5(
  // CHECK: polynomial.add
  // CHECK: polynomial.sub
  // CHECK: return
  func.func @test_ntt_insertion5(%x: !poly_ty_2) -> !poly_ty_2 {
    %a = polynomial.add %x, %x : !poly_ty_2
    %b = polynomial.sub %a, %x : !poly_ty_2
    return %b : !poly_ty_2
  }

  // Covers: coeff-output op (monic_monomial_mul) feeding eval-only MulOp.
  // CHECK: func.func @test_ntt_insertion6(
  // CHECK: polynomial.monic_monomial_mul
  // CHECK: polynomial.ntt
  // CHECK: polynomial.mul
  func.func @test_ntt_insertion6(%x: !poly_ty_1, %k: index) -> !poly_ty_1 {
    %shift = polynomial.monic_monomial_mul %x, %k : (!poly_ty_1, index) -> !poly_ty_1
    %prod = polynomial.mul %shift, %shift : !poly_ty_1
    return %prod : !poly_ty_1
  }

  // Covers: no polynomial inputs; FromTensor producer + MulOp consumer.
  // CHECK: func.func @test_ntt_insertion7(
  // CHECK: polynomial.from_tensor
  // CHECK: polynomial.mul
  func.func @test_ntt_insertion7(%a: i64, %b: i64) -> !poly_ty_3 {
    %coeffs_i64 = tensor.from_elements %a, %b : tensor<2xi64>
    %coeffs = mod_arith.encapsulate %coeffs_i64 : tensor<2xi64> -> tensor<2x!Zq0>
    %p = polynomial.from_tensor %coeffs : tensor<2x!Zq0> -> !poly_ty_3
    %m = polynomial.mul %p, %p : !poly_ty_3
    return %m : !poly_ty_3
  }

  // Covers: no polynomial outputs; coeff-only consumers ToTensor and LeadingTerm.
  // CHECK: func.func @test_ntt_insertion8(
  // CHECK: polynomial.to_tensor
  // CHECK: polynomial.leading_term
  // CHECK: return
  func.func @test_ntt_insertion8(%x: !poly_ty_3) -> (tensor<8x!Zq0>, index, !Zq0) {
    %t = polynomial.to_tensor %x : !poly_ty_3 -> tensor<8x!Zq0>
    %deg, %coeff = polynomial.leading_term %x : !poly_ty_3 -> (index, !Zq0)
    return %t, %deg, %coeff : tensor<8x!Zq0>, index, !Zq0
  }

  // Covers: MulScalar + ModSwitch (flexible ops) followed by eval-only MulOp.
  // CHECK: func.func @test_ntt_insertion9(
  // CHECK: polynomial.mul_scalar
  // CHECK: polynomial.mod_switch
  // CHECK: polynomial.mul
  func.func @test_ntt_insertion9(%x: !poly_ty_3, %s: !Zq0) -> !poly_ty_4 {
    %scaled = polynomial.mul_scalar %x, %s : !poly_ty_3, !Zq0
    %sw = polynomial.mod_switch %scaled : !poly_ty_3 to !poly_ty_4
    %m = polynomial.mul %sw, %sw : !poly_ty_4
    return %m : !poly_ty_4
  }

  // Covers: constant/monomial polynomial producers combined with flexible + eval-only consumers.
  // CHECK: func.func @test_ntt_insertion10(
  // CHECK: polynomial.monomial
  // CHECK: polynomial.constant
  // CHECK: polynomial.add
  // CHECK: polynomial.mul
  func.func @test_ntt_insertion10(%k: index, %c: !Zq0) -> !poly_ty_3 {
    %mono = polynomial.monomial %c, %k : (!Zq0, index) -> !poly_ty_3
    %one = polynomial.constant int<1 + x> : !poly_ty_3
    %sum = polynomial.add %mono, %one : !poly_ty_3
    %prod = polynomial.mul %sum, %one : !poly_ty_3
    return %prod : !poly_ty_3
  }

  // Covers: multiple return statements (including unreachable block) and mixed non-poly outputs.
  // CHECK: func.func @test_ntt_insertion11(
  // CHECK: return
  // CHECK: ^bb1:
  // CHECK: return
  func.func @test_ntt_insertion11(%x: !poly_ty_1) -> (!poly_ty_1, i64) {
    %c0 = arith.constant 0 : i64
    %m0 = polynomial.mul %x, %x : !poly_ty_1
    return %m0, %c0 : !poly_ty_1, i64
  ^bb1:
    %c1 = arith.constant 1 : i64
    %p = polynomial.constant int<1 + x**2> : !poly_ty_1
    %m1 = polynomial.mul %p, %p : !poly_ty_1
    return %m1, %c1 : !poly_ty_1, i64
  }

  // Covers: same value required in both coeff/eval forms, with coeff return tie-break.
  // CHECK: func.func @test_ntt_insertion12([[x13:%.+]]: [[T13C:![^ ]+]]) -> [[T13C]] {
  func.func @test_ntt_insertion12(%x: !poly_ty_1) -> !poly_ty_1 {
    // CHECK-DAG: [[x13e:%.+]] = polynomial.ntt [[x13]] : [[T13C]]
    // CHECK-DAG: %{{.+}} = polynomial.mul [[x13e]], [[x13e]] : [[T13E:![^ ]+]]
    %m = polynomial.mul %x, %x : !poly_ty_1
    // CHECK-DAG: %{{.+}} = polynomial.to_tensor [[x13]] : [[T13C]] -> tensor<1024x{{.+}}>
    %t = polynomial.to_tensor %x : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    // CHECK-DAG: return [[x13]] : [[T13C]]
    return %x : !poly_ty_1
  }

  // Covers: pure declaration with polynomial signature (no body path).
  // CHECK: func.func private @test_ntt_insertion13(!{{.+}}) -> !{{.+}}
  func.func private @test_ntt_insertion13(%x: !poly_ty_1) -> !poly_ty_1

  // Covers: one SSA polynomial value consumed by coeff-only and eval-only ops in one function.
  // CHECK: func.func @test_ntt_insertion14(
  // CHECK: polynomial.add
  // CHECK: polynomial.to_tensor
  // CHECK: polynomial.mul
  func.func @test_ntt_insertion14(%x: !poly_ty_1) -> !poly_ty_1 {
    %v = polynomial.add %x, %x : !poly_ty_1
    %t = polynomial.to_tensor %v : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    %m = polynomial.mul %v, %v : !poly_ty_1
    return %m : !poly_ty_1
  }

  // Covers: baseline MulOp rewrite on tensor<poly> values with form propagation through function signature.
  // CHECK: func.func @test_ntt_insertion15_tensor_mul(
  // CHECK: polynomial.mul
  // CHECK: return
  func.func @test_ntt_insertion15_tensor_mul(%x: tensor<2x!poly_ty_1>) -> tensor<2x!poly_ty_1> {
    %m = polynomial.mul %x, %x : tensor<2x!poly_ty_1>
    return %m : tensor<2x!poly_ty_1>
  }

  // Covers: eval->coeff path for tensor<poly> via MulOp feeding ConvertBasis, which may require tensor INTT.
  // CHECK: func.func @test_ntt_insertion16_tensor_convert_basis(
  // CHECK: polynomial.mul
  // CHECK: polynomial.intt
  // CHECK: polynomial.convert_basis
  // CHECK: return
  func.func @test_ntt_insertion16_tensor_convert_basis(%x: tensor<2x!poly_ty_2>) -> tensor<2x!poly_ty_1> {
    %xsq = polynomial.mul %x, %x : tensor<2x!poly_ty_2>
    %y = polynomial.convert_basis %xsq {targetBasis = !rns.rns<!Zq0>} : tensor<2x!poly_ty_2>
    return %y : tensor<2x!poly_ty_1>
  }

  // Covers: flexible tensor ops (Add/Sub) followed by eval-only tensor MulOp in the same function.
  // CHECK: func.func @test_ntt_insertion17_tensor_flexible(
  // CHECK: polynomial.add
  // CHECK: polynomial.sub
  // CHECK: polynomial.mul
  // CHECK: return
  func.func @test_ntt_insertion17_tensor_flexible(%x: tensor<2x!poly_ty_1>) -> tensor<2x!poly_ty_1> {
    %a = polynomial.add %x, %x : tensor<2x!poly_ty_1>
    %b = polynomial.sub %a, %x : tensor<2x!poly_ty_1>
    %c = polynomial.mul %b, %b : tensor<2x!poly_ty_1>
    return %c : tensor<2x!poly_ty_1>
  }

  // Covers: tensor.from_elements producer of tensor<poly> values consumed by tensor polynomial MulOp.
  // CHECK: func.func @test_ntt_insertion18_tensor_from_elements(
  // CHECK: tensor.from_elements
  // CHECK: polynomial.mul
  // CHECK: return
  func.func @test_ntt_insertion18_tensor_from_elements(%p0: !poly_ty_3, %p1: !poly_ty_3) -> tensor<2x!poly_ty_3> {
    %tp = tensor.from_elements %p0, %p1 : tensor<2x!poly_ty_3>
    %m = polynomial.mul %tp, %tp : tensor<2x!poly_ty_3>
    return %m : tensor<2x!poly_ty_3>
  }

  // Covers: tensor.extract_slice + tensor.extract over tensor<poly> combined with scalar and tensor polynomial ops.
  // CHECK: func.func @test_ntt_insertion19_tensor_extract_paths(
  // CHECK: tensor.extract_slice
  // CHECK: tensor.extract
  // CHECK: polynomial.mul
  // CHECK: tensor.from_elements
  // CHECK: polynomial.add
  // CHECK: return
  func.func @test_ntt_insertion19_tensor_extract_paths(%x: tensor<4x!poly_ty_3>) -> tensor<2x!poly_ty_3> {
    %c0 = arith.constant 0 : index
    %slice = tensor.extract_slice %x[0] [2] [1] : tensor<4x!poly_ty_3> to tensor<2x!poly_ty_3>
    %e0 = tensor.extract %x[%c0] : tensor<4x!poly_ty_3>
    %sq = polynomial.mul %e0, %e0 : !poly_ty_3
    %packed = tensor.from_elements %sq, %sq : tensor<2x!poly_ty_3>
    %sum = polynomial.add %slice, %packed : tensor<2x!poly_ty_3>
    return %sum : tensor<2x!poly_ty_3>
  }
}
