// RUN: heir-opt --bgv-to-polynomial %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

#my_poly = #polynomial.polynomial<1 + x**1024>
#ring1 = #polynomial.ring<cmod=463187969, ideal=#my_poly>
#ring2 = #polynomial.ring<cmod=33538049, ideal=#my_poly>
#rings = #bgv.rings<#ring1, #ring2>
!ct1 = !bgv.ciphertext<rings=#rings, dim=2, level=1>

// CHECK: module
module {
  // CHECK: func.func @test_fn([[X:%.+]]: [[T:tensor<2x!polynomial.*33538049.*]]) -> [[T]] {
  func.func @test_fn(%x : !ct1) -> !ct1 {
    // CHECK: return [[X]] : [[T]]
    return %x : !ct1
  }


  // CHECK: func.func @test_bin_ops([[X:%.+]]: [[T:tensor<2x!polynomial.*33538049.*]], [[Y:%.+]]: [[T]]) {
  func.func @test_bin_ops(%x : !ct1, %y : !ct1) {
    // CHECK: polynomial.add([[X]], [[Y]]) : [[T]]
    %add = bgv.add(%x, %y) : !ct1
    // CHECK: polynomial.sub([[X]], [[Y]]) : [[T]]
    %sub = bgv.sub(%x, %y) : !ct1
    // CHECK: [[C:%.+]] = arith.constant -1 : [[I:.+]]
    // CHECK: polynomial.mul_scalar [[X]], [[C]] : [[T]], [[I]]
    %negate = bgv.negate(%x) : !ct1

    // CHECK: [[I0:%.+]] = arith.constant 0 : index
    // CHECK: [[I1:%.+]] = arith.constant 1 : index
    // CHECK: [[X0:%.+]] = tensor.extract [[X]][[[I0]]] : [[T]]
    // CHECK: [[X1:%.+]] = tensor.extract [[X]][[[I1]]] : [[T]]
    // CHECK: [[Y0:%.+]] = tensor.extract [[Y]][[[I0]]] : [[T]]
    // CHECK: [[Y1:%.+]] = tensor.extract [[Y]][[[I1]]] : [[T]]
    // CHECK: [[Z0:%.+]] = polynomial.mul([[X0]], [[Y0]]) : [[P:!polynomial.*33538049.*]]
    // CHECK: [[X0Y1:%.+]] = polynomial.mul([[X0]], [[Y1]]) : [[P]]
    // CHECK: [[X1Y0:%.+]] = polynomial.mul([[X1]], [[Y0]]) : [[P]]
    // CHECK: [[Z1:%.+]] = polynomial.add([[X0Y1]], [[X1Y0]]) : [[P]]
    // CHECK: [[Z2:%.+]] = polynomial.mul([[X1]], [[Y1]]) : [[P]]
    // CHECK: [[Z:%.+]] = tensor.from_elements [[Z0]], [[Z1]], [[Z2]] : tensor<3x[[P]]>
    %mul = bgv.mul(%x, %y) : !ct1 -> !bgv.ciphertext<rings=#rings, dim=3, level=1>
    return
  }
}
