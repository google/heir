// RUN: heir-opt --lwe-to-cheddar-gpu %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>

!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329], P = [72057594038321153], logDefaultScale = 40>} {
  // CHECK: func.func @passthrough(%[[OUT_BX:[^,]+]]: memref<3x65536xi64>, %[[OUT_AX:[^,]+]]: memref<3x65536xi64>, %[[IN_BX:[^,]+]]: memref<3x65536xi64>, %[[IN_AX:[^)]+]]: memref<3x65536xi64>)
  // CHECK-SAME: attributes {{{.*}}cheddar_gpu.abi = {{.*}}llvm.emit_c_interface{{.*}}} {
  // CHECK-DAG:  cheddar_gpu.copy %[[OUT_BX]], %[[IN_BX]] : memref<3x65536xi64>, memref<3x65536xi64>
  // CHECK-DAG:  cheddar_gpu.copy %[[OUT_AX]], %[[IN_AX]] : memref<3x65536xi64>, memref<3x65536xi64>
  // CHECK:      return
  // CHECK: }
  func.func @passthrough(%arg0: !ct_L2) -> !ct_L2 {
    return %arg0 : !ct_L2
  }
}
