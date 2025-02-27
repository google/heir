//RUN: heir-opt --polynomial-to-pisa %s | FileCheck %s

!coeff_ty = !mod_arith.int<33538049:i32>
!p = !polynomial.polynomial<ring=<coefficientType=!coeff_ty, polynomialModulus=#polynomial.int_polynomial<1 + x**8192>>>

//CHECK-LABEL: @test_add
//CHECK: [[X:%.+]]: tensor<8192x!Z33538049_i32_>, [[Y:%.+]]: tensor<8192x!Z33538049_i32_>
func.func @test_add(%x : !p, %y : !p) -> !p {
    //CHECK: [[ADD:%.+]] = pisa.add [[X]], [[Y]] {i = 0 : i32, q = 33538049 : i32} : tensor<8192x!Z33538049_i32_>
    %0 = polynomial.add %x, %y : !p
    //CHECK: return [[ADD]] : tensor<8192x!Z33538049_i32_>
    return %0 : !p
}
