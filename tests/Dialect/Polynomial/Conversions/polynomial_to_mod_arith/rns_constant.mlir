// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

!Z17 = !mod_arith.int<17 : i32>
!Z19 = !mod_arith.int<19 : i32>
!rns = !rns.rns<!Z17, !Z19>
#ring = #polynomial.ring<coefficientType = !rns, polynomialModulus = <1 + x**4>>
!poly = !polynomial.polynomial<ring = #ring>

// CHECK: func.func @test_rns_constant
func.func @test_rns_constant() -> !poly {
  // -1 reduces to 16 modulo 17 and 18 modulo 19.
  // CHECK: %[[STORAGE:.*]] = arith.constant dense<{{\[\[16, 18\], \[2, 2\], \[0, 0\], \[1, 1\]\]}}> : tensor<4x2xi32>
  // CHECK: %[[CONSTANT:.*]] = mod_arith.encapsulate %[[STORAGE]] : tensor<4x2xi32> -> tensor<4x!rns.rns<{{.*}}>>
  %0 = polynomial.constant int<-1 + 2x + x**3> : !poly
  // CHECK: return %[[CONSTANT]]
  return %0 : !poly
}

// CHECK: func.func @test_rns_polynomial_attr_constant
func.func @test_rns_polynomial_attr_constant() -> !poly {
  // The attribute stores residues limb-major (numLimbs x numCoeffs), while the
  // lowered storage is coefficient-major (degree x numLimbs) and zero-padded to
  // the ring degree.
  // CHECK: %[[STORAGE:.*]] = arith.constant dense<{{\[\[1, 3\], \[2, 4\], \[0, 0\], \[0, 0\]\]}}> : tensor<4x2xi32>
  // CHECK: %[[CONSTANT:.*]] = mod_arith.encapsulate %[[STORAGE]] : tensor<4x2xi32> -> tensor<4x!rns.rns<{{.*}}>>
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>> : !poly
  // CHECK: return %[[CONSTANT]]
  return %0 : !poly
}

!poly_eval = !polynomial.polynomial<ring = #ring, form = eval>

// CHECK: func.func @test_rns_polynomial_attr_constant_eval
func.func @test_rns_polynomial_attr_constant_eval() -> !poly_eval {
  // CHECK: %[[STORAGE:.*]] = arith.constant dense<{{\[\[1, 5\], \[2, 6\], \[3, 7\], \[4, 8\]\]}}> : tensor<4x2xi32>
  // CHECK: %[[CONSTANT:.*]] = mod_arith.encapsulate %[[STORAGE]] : tensor<4x2xi32> -> tensor<4x!rns.rns<{{.*}}>>
  %0 = polynomial.constant #polynomial.rns_polynomial<dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>, eval> : !poly_eval
  // CHECK: return %[[CONSTANT]]
  return %0 : !poly_eval
}
