// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

!Z0 = !mod_arith.int<17 : i64>
!Z1 = !mod_arith.int<41 : i64>
!rns = !rns.rns<!Z0, !Z1>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**8>>
!poly = !polynomial.polynomial<ring = #ring>

#v0 = #mod_arith.value<9 : !Z0>
#v1 = #mod_arith.value<38 : !Z1>
#rns_root = #rns.value<[#v0, #v1]>
#root = #polynomial.primitive_root<value = #rns_root, degree = 8 : i32>

!ntt_poly = !polynomial.polynomial<ring = #ring, form = eval>

// CHECK: [[Z17:!.*]] = !mod_arith.int<17 : i64>
// CHECK: [[Z41:!.*]] = !mod_arith.int<41 : i64>
// CHECK: [[RNS:!.*]] = !rns.rns<[[Z17]], [[Z41]]>

func.func @test_ntt_rns(%p0 : !poly) -> !ntt_poly {
  // CHECK-NOT: polynomial.ntt
  // CHECK: %[[ROOTS_STORAGE:.*]] = arith.constant dense<{{.*}}> : tensor<8x2xi64>
  // CHECK: %[[ROOTS:.*]] = mod_arith.encapsulate %[[ROOTS_STORAGE]] : tensor<8x2xi64> -> tensor<8x[[RNS]]>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: mod_arith.mul
  // CHECK: mod_arith.add
  // CHECK: mod_arith.sub
  %1 = polynomial.ntt %p0 {root = #root} : !poly
  return %1 : !ntt_poly
}
