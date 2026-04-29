// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s
// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s --check-prefixes=CHECK,ORDER
// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s --check-prefixes=CHECK,NTT_PLACEMENT

// CHECK-DAG: [[ZQ0:![^ ]+]] = !mod_arith.int<1095233372161 : i64>
!Zq0 = !mod_arith.int<1095233372161 : i64>
// CHECK-DAG: [[ZQ1:![^ ]+]] = !mod_arith.int<1032955396097 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
// CHECK-DAG: [[RNS1:![^ ]+]] = !rns.rns<[[ZQ0]]>
// CHECK-DAG: [[ring_1:#[^ ]+]] = #polynomial.ring<coefficientType = [[RNS1]], polynomialModulus = <1 + x**1024>>
#ring_1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
// CHECK-DAG: [[poly_ty_1:![^ ]+]] = !polynomial.polynomial<ring = [[ring_1]]>
!poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=coeff>
// CHECK-DAG: [[ntt_poly_ty_1:![^ ]+]] = !polynomial.polynomial<ring = [[ring_1]], form = eval>
!ntt_poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=eval>

// CHECK-DAG: [[RNS2:![^ ]+]] = !rns.rns<[[ZQ0]], [[ZQ1]]>
// CHECK-DAG: [[ring_2:#[^ ]+]] = #polynomial.ring<coefficientType = [[RNS2]], polynomialModulus = <1 + x**1024>>
#ring_2 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1>, polynomialModulus = <1 + x**1024>>
// CHECK-DAG: [[poly_ty_2:![^ ]+]] = !polynomial.polynomial<ring = [[ring_2]]>
!poly_ty_2 = !polynomial.polynomial<ring=#ring_2, form=coeff>
// CHECK-DAG: [[ntt_poly_ty_2:![^ ]+]] = !polynomial.polynomial<ring = [[ring_2]], form = eval>
!ntt_poly_ty_2 = !polynomial.polynomial<ring=#ring_2, form=eval>

// CHECK-DAG: [[ring_3:#[^ ]+]] = #polynomial.ring<coefficientType = [[ZQ0]], polynomialModulus = <1 + x**8>>
#ring_3 = #polynomial.ring<coefficientType = !Zq0, polynomialModulus = <1 + x**8>>
// CHECK-DAG: [[poly_ty_3:![^ ]+]] = !polynomial.polynomial<ring = [[ring_3]]>
!poly_ty_3 = !polynomial.polynomial<ring=#ring_3, form=coeff>
// CHECK-DAG: [[ntt_poly_ty_3:![^ ]+]] = !polynomial.polynomial<ring = [[ring_3]], form = eval>
!ntt_poly_ty_3 = !polynomial.polynomial<ring=#ring_3, form=eval>

// CHECK-DAG: [[ring_4:#[^ ]+]] = #polynomial.ring<coefficientType = [[ZQ1]], polynomialModulus = <1 + x**8>>
#ring_4 = #polynomial.ring<coefficientType = !Zq1, polynomialModulus = <1 + x**8>>
!poly_ty_4 = !polynomial.polynomial<ring=#ring_4, form=coeff>
// CHECK-DAG: [[ntt_poly_ty_4:![^ ]+]] = !polynomial.polynomial<ring = [[ring_4]], form = eval>
!ntt_poly_ty_4 = !polynomial.polynomial<ring=#ring_4, form=eval>

module {
  // Covers: baseline MulOp rewrite from coeff-form input to eval-form signature/result.
  // CHECK: func.func @test_ntt_insertion1([[x:%.+]]: [[ntt_poly_ty_1]]) -> [[ntt_poly_ty_1]] {
  func.func @test_ntt_insertion1(%x: !poly_ty_1) -> !poly_ty_1 {
    // CHECK: [[xsq:%.+]] = polynomial.mul [[x]], [[x]] : [[ntt_poly_ty_1]]
    %xsq = polynomial.mul %x, %x : !poly_ty_1
    // CHECK: return [[xsq]] : [[ntt_poly_ty_1]]
    return %xsq: !poly_ty_1
  }

  // Covers: eval->coeff path with INTT + apply_coefficientwise to satisfy narrower output basis.
  // CHECK: func.func @test_ntt_insertion2([[x2:%.+]]: [[ntt_poly_ty_2]]) -> [[poly_ty_1]] {
  func.func @test_ntt_insertion2(%x: !poly_ty_2) -> !poly_ty_1 {
    // CHECK: [[xsq2:%.+]] = polynomial.mul [[x2]], [[x2]] : [[ntt_poly_ty_2]]
    %xsq = polynomial.mul %x, %x : !poly_ty_2
    // CHECK: [[coeff2:%.+]] = polynomial.intt [[xsq2]] : [[ntt_poly_ty_2]]
    // CHECK: [[out2:%.+]] = polynomial.apply_coefficientwise{{ *}}([[coeff2]] : [[poly_ty_2]])
    %y = polynomial.apply_coefficientwise (%xsq : !poly_ty_2) {
    ^body(%coeff: !rns.rns<!Zq0, !Zq1>, %degree: index):
      %reduced = rns.extract_slice %coeff {start = 0 : index, size = 1 : index} : !rns.rns<!Zq0, !Zq1> -> !rns.rns<!Zq0>
      polynomial.yield %reduced : !rns.rns<!Zq0>
    } -> !poly_ty_1
    // CHECK: return [[out2]] : [[poly_ty_1]]
    return %y: !poly_ty_1
  }

  // Covers: flexible Add/Sub mode + forced eval by downstream MulOp in same function.
  // NTT_PLACEMENT: func.func @test_ntt_insertion3([[x3:%.+]]: [[ntt_poly_ty_1]]) -> [[ntt_poly_ty_1]] {
  // ORDER: func.func @test_ntt_insertion3([[x3:%.+]]: [[ntt_poly_ty_1]]) -> [[ntt_poly_ty_1]] {
  func.func @test_ntt_insertion3(%x: !poly_ty_1) -> !poly_ty_1 {
    // NTT_PLACEMENT-NOT: polynomial.ntt
    // ORDER: [[a3:%.+]] = polynomial.add {{.*}} : [[ntt_poly_ty_1]]
    %a = polynomial.add %x, %x : !poly_ty_1
    // ORDER: [[b3:%.+]] = polynomial.sub [[a3]], {{.*}} : [[ntt_poly_ty_1]]
    %b = polynomial.sub %a, %x : !poly_ty_1
    // NTT_PLACEMENT: [[c3:%.+]] = polynomial.mul {{.*}} : [[ntt_poly_ty_1]]
    // ORDER: [[c3:%.+]] = polynomial.mul [[b3]], [[b3]] : [[ntt_poly_ty_1]]
    %c = polynomial.mul %b, %b : !poly_ty_1
    // NTT_PLACEMENT: return [[c3]] : [[ntt_poly_ty_1]]
    // ORDER: return [[c3]] : [[ntt_poly_ty_1]]
    return %c : !poly_ty_1
  }

  // Covers: coeff-output op (monic_monomial_mul) feeding eval-only MulOp.
  // CHECK: func.func @test_ntt_insertion6([[x6:%.+]]: [[poly_ty_1]], [[k6:%.+]]: index) -> [[ntt_poly_ty_1]] {
  // CHECK: [[shift6:%.+]] = polynomial.monic_monomial_mul [[x6]], [[k6]] : ([[poly_ty_1]], index) -> [[poly_ty_1]]
  // CHECK: [[shift6e:%.+]] = polynomial.ntt [[shift6]] : [[poly_ty_1]]
  // CHECK: [[prod6:%.+]] = polynomial.mul [[shift6e]], [[shift6e]] : [[ntt_poly_ty_1]]
  // CHECK: return [[prod6]] : [[ntt_poly_ty_1]]
  func.func @test_ntt_insertion6(%x: !poly_ty_1, %k: index) -> !poly_ty_1 {
    %shift = polynomial.monic_monomial_mul %x, %k : (!poly_ty_1, index) -> !poly_ty_1
    %prod = polynomial.mul %shift, %shift : !poly_ty_1
    return %prod : !poly_ty_1
  }

  // Covers: no polynomial inputs; FromTensor producer + MulOp consumer.
  // CHECK: func.func @test_ntt_insertion7([[a7:%.+]]: i64, [[b7:%.+]]: i64) -> [[ntt_poly_ty_3]] {
  // CHECK: [[p7:%.+]] = polynomial.from_tensor {{.*}} -> [[poly_ty_3]]
  // CHECK: [[m7:%.+]] = polynomial.mul {{.*}} : [[ntt_poly_ty_3]]
  // CHECK: return [[m7]] : [[ntt_poly_ty_3]]
  func.func @test_ntt_insertion7(%a: i64, %b: i64) -> !poly_ty_3 {
    %coeffs_i64 = tensor.from_elements %a, %b : tensor<2xi64>
    %coeffs = mod_arith.encapsulate %coeffs_i64 : tensor<2xi64> -> tensor<2x!Zq0>
    %p = polynomial.from_tensor %coeffs : tensor<2x!Zq0> -> !poly_ty_3
    %m = polynomial.mul %p, %p : !poly_ty_3
    return %m : !poly_ty_3
  }

  // Covers: no polynomial outputs; coeff-only consumers ToTensor and LeadingTerm.
  // CHECK: func.func @test_ntt_insertion8([[x8:%.+]]: [[poly_ty_3]]) -> (tensor<8x[[ZQ0]]>, index, [[ZQ0]]) {
  // CHECK: [[t8:%.+]] = polynomial.to_tensor [[x8]] : [[poly_ty_3]] -> tensor<8x[[ZQ0]]>
  // CHECK: [[deg8:%.+]], [[coeff8:%.+]] = polynomial.leading_term [[x8]] : [[poly_ty_3]] -> (index, [[ZQ0]])
  // CHECK: return [[t8]], [[deg8]], [[coeff8]] : tensor<8x[[ZQ0]]>, index, [[ZQ0]]
  func.func @test_ntt_insertion8(%x: !poly_ty_3) -> (tensor<8x!Zq0>, index, !Zq0) {
    %t = polynomial.to_tensor %x : !poly_ty_3 -> tensor<8x!Zq0>
    %deg, %coeff = polynomial.leading_term %x : !poly_ty_3 -> (index, !Zq0)
    return %t, %deg, %coeff : tensor<8x!Zq0>, index, !Zq0
  }

  // Covers: MulScalar + ModSwitch (flexible ops) followed by eval-only MulOp.
  // CHECK: func.func @test_ntt_insertion9([[x9:%.+]]: [[ntt_poly_ty_3]], [[s9:%.+]]: [[ZQ0]]) -> [[ntt_poly_ty_4]] {
  // CHECK-NOT: polynomial.ntt
  // CHECK: [[scaled9:%.+]] = polynomial.mul_scalar [[x9]], [[s9]] : [[ntt_poly_ty_3]], [[ZQ0]]
  // CHECK: [[sw9:%.+]] = polynomial.mod_switch [[scaled9]] : [[ntt_poly_ty_3]] to [[ntt_poly_ty_4]]
  // CHECK: [[m9:%.+]] = polynomial.mul [[sw9]], [[sw9]] : [[ntt_poly_ty_4]]
  // CHECK: return [[m9]] : [[ntt_poly_ty_4]]
  func.func @test_ntt_insertion9(%x: !poly_ty_3, %s: !Zq0) -> !poly_ty_4 {
    %scaled = polynomial.mul_scalar %x, %s : !poly_ty_3, !Zq0
    %sw = polynomial.mod_switch %scaled : !poly_ty_3 to !poly_ty_4
    %m = polynomial.mul %sw, %sw : !poly_ty_4
    return %m : !poly_ty_4
  }

  // Covers: constant/monomial polynomial producers combined with flexible + eval-only consumers.
  // CHECK: func.func @test_ntt_insertion10([[k10:%.+]]: index, [[c10:%.+]]: [[ZQ0]]) -> [[ntt_poly_ty_3]] {
  // CHECK-NOT: polynomial.ntt
  // CHECK: [[mono10:%.+]] = polynomial.monomial [[c10]], [[k10]] : ([[ZQ0]], index) -> [[ntt_poly_ty_3]]
  // CHECK: [[one10:%.+]] = polynomial.constant int<1 + x> : [[ntt_poly_ty_3]]
  // CHECK: [[sum10:%.+]] = polynomial.add [[mono10]], [[one10]] : [[ntt_poly_ty_3]]
  // CHECK: [[prod10:%.+]] = polynomial.mul [[sum10]], [[one10]] : [[ntt_poly_ty_3]]
  // CHECK: return [[prod10]] : [[ntt_poly_ty_3]]
  func.func @test_ntt_insertion10(%k: index, %c: !Zq0) -> !poly_ty_3 {
    %mono = polynomial.monomial %c, %k : (!Zq0, index) -> !poly_ty_3
    %one = polynomial.constant int<1 + x> : !poly_ty_3
    %sum = polynomial.add %mono, %one : !poly_ty_3
    %prod = polynomial.mul %sum, %one : !poly_ty_3
    return %prod : !poly_ty_3
  }

  // Covers: multiple return statements (including unreachable block) and mixed non-poly outputs.
  // CHECK: func.func @test_ntt_insertion11([[x11:%.+]]: [[ntt_poly_ty_1]]) -> ([[ntt_poly_ty_1]], i64) {
  // CHECK: [[m0_11:%.+]] = polynomial.mul [[x11]], [[x11]] : [[ntt_poly_ty_1]]
  // CHECK: return [[m0_11]], %{{.+}} : [[ntt_poly_ty_1]], i64
  // CHECK: ^bb1:
  // CHECK: [[p11:%.+]] = polynomial.constant int<1 + x**2> : [[ntt_poly_ty_1]]
  // CHECK: [[m1_11:%.+]] = polynomial.mul [[p11]], [[p11]] : [[ntt_poly_ty_1]]
  // CHECK: return [[m1_11]], %{{.+}} : [[ntt_poly_ty_1]], i64
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

  // Covers: dead code
  // CHECK: func.func @test_ntt_insertion12([[x13:%.+]]: [[poly_ty_1]]) -> [[poly_ty_1]] {
  func.func @test_ntt_insertion12(%x: !poly_ty_1) -> !poly_ty_1 {
    // CHECK-NOT: polynomial.ntt
    // CHECK-NOT: polynomial.intt
    // CHECK-NOT: polynomial.mul
    // CHECK-NOT: polynomial.to_tensor
    %m = polynomial.mul %x, %x : !poly_ty_1
    %t = polynomial.to_tensor %x : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    // CHECK: return [[x13]] : [[poly_ty_1]]
    return %x : !poly_ty_1
  }

  // Covers: same value required in both coeff/eval forms, with coeff return tie-break.
  // CHECK: func.func @test_ntt_insertion13([[x13:%.+]]: [[poly_ty_1]]) -> ([[poly_ty_1]], [[ntt_poly_ty_1]], tensor<1024x{{.*}}>) {
  func.func @test_ntt_insertion13(%x: !poly_ty_1) -> (!poly_ty_1, !poly_ty_1, tensor<1024x!rns.rns<!Zq0>>) {
    // CHECK: [[x13e:%.+]] = polynomial.ntt [[x13]] : [[poly_ty_1]]
    // CHECK: %{{.+}} = polynomial.mul [[x13e]], [[x13e]] : [[ntt_poly_ty_1]]
    %m = polynomial.mul %x, %x : !poly_ty_1
    %t = polynomial.to_tensor %x : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    // CHECK: return [[x13]], {{.*}}, {{.*}} : [[poly_ty_1]], [[ntt_poly_ty_1]], tensor<1024x{{.*}}>
    return %x, %m, %t : !poly_ty_1, !poly_ty_1, tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: one SSA polynomial value consumed by coeff-only and eval-only ops in one function.
  // CHECK: func.func @test_ntt_insertion14([[x14:%.+]]: [[poly_ty_1]]) -> ([[ntt_poly_ty_1]], tensor<1024x{{.*}}>) {
  // CHECK: [[v14:%.+]] = polynomial.add [[x14]], [[x14]] : [[poly_ty_1]]
  // CHECK: %{{.+}} = polynomial.to_tensor [[v14]] : [[poly_ty_1]] -> tensor<1024x[[RNS1]]>
  // CHECK: [[m14:%.+]] = polynomial.mul {{.*}} : [[ntt_poly_ty_1]]
  // CHECK: return [[m14]], {{.*}} : [[ntt_poly_ty_1]], tensor<1024x{{.*}}>
  func.func @test_ntt_insertion14(%x: !poly_ty_1) -> (!poly_ty_1, tensor<1024x!rns.rns<!Zq0>>) {
    %v = polynomial.add %x, %x : !poly_ty_1
    %t = polynomial.to_tensor %v : !poly_ty_1 -> tensor<1024x!rns.rns<!Zq0>>
    %m = polynomial.mul %v, %v : !poly_ty_1
    return %m, %t : !poly_ty_1, tensor<1024x!rns.rns<!Zq0>>
  }

  // Covers: baseline MulOp rewrite on tensor<poly> values with form propagation through function signature.
  // CHECK: func.func @test_ntt_insertion15_tensor_mul([[x15:%.+]]: tensor<2x[[ntt_poly_ty_1]]>) -> tensor<2x[[ntt_poly_ty_1]]> {
  // CHECK: [[m15:%.+]] = polynomial.mul [[x15]], [[x15]] : tensor<2x[[ntt_poly_ty_1]]>
  // CHECK: return [[m15]] : tensor<2x[[ntt_poly_ty_1]]>
  func.func @test_ntt_insertion15_tensor_mul(%x: tensor<2x!poly_ty_1>) -> tensor<2x!poly_ty_1> {
    %m = polynomial.mul %x, %x : tensor<2x!poly_ty_1>
    return %m : tensor<2x!poly_ty_1>
  }

  // Covers: eval->coeff path for tensor<poly> via MulOp feeding apply_coefficientwise, which may require tensor INTT.
  // CHECK: func.func @test_ntt_insertion16_tensor_convert_basis([[x16:%.+]]: tensor<2x[[ntt_poly_ty_2]]>) -> tensor<2x[[poly_ty_1]]> {
  // CHECK: [[xsq16:%.+]] = polynomial.mul [[x16]], [[x16]] : tensor<2x[[ntt_poly_ty_2]]>
  // CHECK: [[coeff16:%.+]] = polynomial.intt [[xsq16]] : tensor<2x[[ntt_poly_ty_2]]>
  // CHECK: [[coeff16_0:%.+]] = tensor.extract [[coeff16]][%{{.+}}] : tensor<2x[[poly_ty_2]]>
  // CHECK: [[coeff16_1:%.+]] = tensor.extract [[coeff16]][%{{.+}}] : tensor<2x[[poly_ty_2]]>
  // CHECK: [[y16_0:%.+]] = polynomial.apply_coefficientwise{{ *}}([[coeff16_0]] : [[poly_ty_2]])
  // CHECK: [[y16_1:%.+]] = polynomial.apply_coefficientwise{{ *}}([[coeff16_1]] : [[poly_ty_2]])
  // CHECK: [[y16:%.+]] = tensor.from_elements [[y16_0]], [[y16_1]] : tensor<2x[[poly_ty_1]]>
  // CHECK: return [[y16]] : tensor<2x[[poly_ty_1]]>
  func.func @test_ntt_insertion16_tensor_convert_basis(%x: tensor<2x!poly_ty_2>) -> tensor<2x!poly_ty_1> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %xsq = polynomial.mul %x, %x : tensor<2x!poly_ty_2>
    %xsq0 = tensor.extract %xsq[%c0] : tensor<2x!poly_ty_2>
    %xsq1 = tensor.extract %xsq[%c1] : tensor<2x!poly_ty_2>
    %y0 = polynomial.apply_coefficientwise (%xsq0 : !poly_ty_2) {
    ^body(%coeff: !rns.rns<!Zq0, !Zq1>, %degree: index):
      %reduced = rns.extract_slice %coeff {start = 0 : index, size = 1 : index} : !rns.rns<!Zq0, !Zq1> -> !rns.rns<!Zq0>
      polynomial.yield %reduced : !rns.rns<!Zq0>
    } -> !poly_ty_1
    %y1 = polynomial.apply_coefficientwise (%xsq1 : !poly_ty_2) {
    ^body(%coeff: !rns.rns<!Zq0, !Zq1>, %degree: index):
      %reduced = rns.extract_slice %coeff {start = 0 : index, size = 1 : index} : !rns.rns<!Zq0, !Zq1> -> !rns.rns<!Zq0>
      polynomial.yield %reduced : !rns.rns<!Zq0>
    } -> !poly_ty_1
    %y = tensor.from_elements %y0, %y1 : tensor<2x!poly_ty_1>
    return %y : tensor<2x!poly_ty_1>
  }

  // Covers: flexible tensor ops (Add/Sub) followed by eval-only tensor MulOp in the same function.
  // CHECK: func.func @test_ntt_insertion17_tensor_flexible([[x17:%.+]]: tensor<2x[[ntt_poly_ty_1]]>) -> tensor<2x[[ntt_poly_ty_1]]> {
  // CHECK-NOT: polynomial.ntt
  // CHECK: [[a17:%.+]] = polynomial.add [[x17]], [[x17]] : tensor<2x[[ntt_poly_ty_1]]>
  // CHECK: [[b17:%.+]] = polynomial.sub [[a17]], [[x17]] : tensor<2x[[ntt_poly_ty_1]]>
  // CHECK: [[c17:%.+]] = polynomial.mul [[b17]], [[b17]] : tensor<2x[[ntt_poly_ty_1]]>
  // CHECK: return [[c17]] : tensor<2x[[ntt_poly_ty_1]]>
  func.func @test_ntt_insertion17_tensor_flexible(%x: tensor<2x!poly_ty_1>) -> tensor<2x!poly_ty_1> {
    %a = polynomial.add %x, %x : tensor<2x!poly_ty_1>
    %b = polynomial.sub %a, %x : tensor<2x!poly_ty_1>
    %c = polynomial.mul %b, %b : tensor<2x!poly_ty_1>
    return %c : tensor<2x!poly_ty_1>
  }

  // Covers: tensor.from_elements producer of tensor<poly> values consumed by tensor polynomial MulOp.
  // CHECK: func.func @test_ntt_insertion18_tensor_from_elements([[p0_18:%.+]]: [[ntt_poly_ty_3]], [[p1_18:%.+]]: [[ntt_poly_ty_3]]) -> tensor<2x[[ntt_poly_ty_3]]> {
  // CHECK-NOT: polynomial.ntt
  // CHECK: [[tp18:%.+]] = tensor.from_elements [[p0_18]], [[p1_18]] : tensor<2x[[ntt_poly_ty_3]]>
  // CHECK: [[m18:%.+]] = polynomial.mul [[tp18]], [[tp18]] : tensor<2x[[ntt_poly_ty_3]]>
  // CHECK: return [[m18]] : tensor<2x[[ntt_poly_ty_3]]>
  func.func @test_ntt_insertion18_tensor_from_elements(%p0: !poly_ty_3, %p1: !poly_ty_3) -> tensor<2x!poly_ty_3> {
    %tp = tensor.from_elements %p0, %p1 : tensor<2x!poly_ty_3>
    %m = polynomial.mul %tp, %tp : tensor<2x!poly_ty_3>
    return %m : tensor<2x!poly_ty_3>
  }

  // Covers: tensor.extract_slice + tensor.extract over tensor<poly> combined with scalar and tensor polynomial ops.
  // CHECK: func.func @test_ntt_insertion19_tensor_extract_paths([[x19:%.+]]: tensor<4x[[ntt_poly_ty_3]]>) -> tensor<2x[[ntt_poly_ty_3]]> {
  // CHECK-NOT: polynomial.ntt
  // CHECK: [[slice19:%.+]] = tensor.extract_slice [[x19]][0] [2] [1] : tensor<4x[[ntt_poly_ty_3]]> to tensor<2x[[ntt_poly_ty_3]]>
  // CHECK: [[e19:%.+]] = tensor.extract [[x19]][%{{.+}}] : tensor<4x[[ntt_poly_ty_3]]>
  // CHECK: [[sq19:%.+]] = polynomial.mul [[e19]], [[e19]] : [[ntt_poly_ty_3]]
  // CHECK: [[packed19:%.+]] = tensor.from_elements [[sq19]], [[sq19]] : tensor<2x[[ntt_poly_ty_3]]>
  // CHECK: [[sum19:%.+]] = polynomial.add [[slice19]], [[packed19]] : tensor<2x[[ntt_poly_ty_3]]>
  // CHECK: return [[sum19]] : tensor<2x[[ntt_poly_ty_3]]>
  func.func @test_ntt_insertion19_tensor_extract_paths(%x: tensor<4x!poly_ty_3>) -> tensor<2x!poly_ty_3> {
    %c0 = arith.constant 0 : index
    %slice = tensor.extract_slice %x[0] [2] [1] : tensor<4x!poly_ty_3> to tensor<2x!poly_ty_3>
    %e0 = tensor.extract %x[%c0] : tensor<4x!poly_ty_3>
    %sq = polynomial.mul %e0, %e0 : !poly_ty_3
    %packed = tensor.from_elements %sq, %sq : tensor<2x!poly_ty_3>
    %sum = polynomial.add %slice, %packed : tensor<2x!poly_ty_3>
    return %sum : tensor<2x!poly_ty_3>
  }

  // Covers: constant needed in both coeff and eval forms in one function.
  // CHECK: func.func @test_ntt_insertion20_constant_needed_in_both_forms() -> ([[ntt_poly_ty_3]], tensor<8x[[ZQ0]]>) {
  // CHECK-NOT: polynomial.ntt
  // CHECK: [[c20:%.+]] = polynomial.constant int<1 + x> : [[poly_ty_3]]{{$}}
  // CHECK: [[c20e:%.+]] = polynomial.constant int<1 + x> : [[ntt_poly_ty_3]]{{$}}
  // CHECK: [[m20:%.+]] = polynomial.mul [[c20e]], [[c20e]] : [[ntt_poly_ty_3]]
  // CHECK: [[t20:%.+]] = polynomial.to_tensor [[c20]] : [[poly_ty_3]] -> tensor<8x[[ZQ0]]>
  // CHECK: return [[m20]], [[t20]] : [[ntt_poly_ty_3]], tensor<8x[[ZQ0]]>
  func.func @test_ntt_insertion20_constant_needed_in_both_forms() -> (!poly_ty_3, tensor<8x!Zq0>) {
    %c = polynomial.constant int<1 + x> : !poly_ty_3
    %m = polynomial.mul %c, %c : !poly_ty_3
    %t = polynomial.to_tensor %c : !poly_ty_3 -> tensor<8x!Zq0>
    return %m, %t : !poly_ty_3, tensor<8x!Zq0>
  }
}
