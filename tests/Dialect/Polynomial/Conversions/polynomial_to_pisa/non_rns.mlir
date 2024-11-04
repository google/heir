//RUN: heir-opt --polynomial-to-pisa %s | FileCheck %s

#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 33538049 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**8192>>
!p = !polynomial.polynomial<ring=#ring>

//CHECK-LABEL: @test_add
//CHECK: [[X:%.+]]: tensor<8192xi32>, [[Y:%.+]]: tensor<8192xi32>
func.func @test_add(%x : !p, %y : !p) -> !p {
    //CHECK: [[ADD:%.+]] = pisa.add [[X]], [[Y]] {i = 0 : i32, q = 33538049 : i32} : tensor<8192xi32>
    %0 = polynomial.add %x, %y : !p
    //CHECK: return [[ADD]] : tensor<8192xi32>
    return %0 : !p
}
