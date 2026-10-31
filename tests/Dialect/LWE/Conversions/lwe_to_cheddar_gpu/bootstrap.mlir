// RUN: heir-opt --lwe-to-cheddar-gpu="params=%S/bootparam_40_64bit.yaml" %s | FileCheck %s
// RUN: heir-opt --scheme-to-cheddar-gpu="params=%S/bootparam_40_64bit.yaml" %s | FileCheck %s --check-prefix=PIPELINE

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 40>
#key = #lwe.key<>
#ring_f64_1_x65536 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**65536>>

!rns_L0 = !rns.rns<!mod_arith.int<1125899908022273 : i64>>
#ring_rns_L0 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0, encryption_type = mix>
#chain_L0 = #lwe.modulus_chain<elements = <1125899908022273 : i64>, current = 0>
!ct_L0 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #chain_L0>

!rns_L2 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>>
#ring_rns_L2 = #polynomial.ring<coefficientType = !rns_L2, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L2 = #lwe.ciphertext_space<ring = #ring_rns_L2, encryption_type = mix>
#chain_L2 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64>, current = 2>
!ct_L2 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L2, key = #key, modulus_chain = #chain_L2>

!rns_L8 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>, !mod_arith.int<1099525128193 : i64>, !mod_arith.int<1099526176769 : i64>, !mod_arith.int<1099529060353 : i64>, !mod_arith.int<1099535220737 : i64>, !mod_arith.int<1099536138241 : i64>, !mod_arith.int<1099537580033 : i64>>
#ring_rns_L8 = #polynomial.ring<coefficientType = !rns_L8, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L8 = #lwe.ciphertext_space<ring = #ring_rns_L8, encryption_type = mix>
#chain_L8 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64, 1099525128193 : i64, 1099526176769 : i64, 1099529060353 : i64, 1099535220737 : i64, 1099536138241 : i64, 1099537580033 : i64>, current = 8>
!ct_L8 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L8, key = #key, modulus_chain = #chain_L8>

!rns_L10 = !rns.rns<!mod_arith.int<1125899908022273 : i64>, !mod_arith.int<1099515691009 : i64>, !mod_arith.int<1099523555329 : i64>, !mod_arith.int<1099525128193 : i64>, !mod_arith.int<1099526176769 : i64>, !mod_arith.int<1099529060353 : i64>, !mod_arith.int<1099535220737 : i64>, !mod_arith.int<1099536138241 : i64>, !mod_arith.int<1099537580033 : i64>, !mod_arith.int<1099538104321 : i64>, !mod_arith.int<1099540725761 : i64>>
#ring_rns_L10 = #polynomial.ring<coefficientType = !rns_L10, polynomialModulus = <1 + x**65536>>
#ciphertext_space_L10 = #lwe.ciphertext_space<ring = #ring_rns_L10, encryption_type = mix>
#chain_L10 = #lwe.modulus_chain<elements = <1125899908022273 : i64, 1099515691009 : i64, 1099523555329 : i64, 1099525128193 : i64, 1099526176769 : i64, 1099529060353 : i64, 1099535220737 : i64, 1099536138241 : i64, 1099537580033 : i64, 1099538104321 : i64, 1099540725761 : i64>, current = 10>
!ct_L10 = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x65536, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L10, key = #key, modulus_chain = #chain_L10>

// CHECK: module attributes {
// CHECK-SAME: cheddar_gpu.oracle_trace = "{{.*}}bootstrap{{.*}}"

// CHECK: func.func @boot_default_full_slot(%[[OUT_BX:[^,]+]]: memref<11x65536xi64>, %[[OUT_AX:[^,]+]]: memref<11x65536xi64>, %[[IN_BX:[^,]+]]: memref<1x65536xi64>, %[[IN_AX:[^,]+]]: memref<1x65536xi64>, %[[CONSTS:[^,]+]]: memref<?xi8>, %[[EVK:[^)]+]]: memref<?xi8>)
// CHECK-SAME: cheddar_gpu.abi = {args = [{{.*}}], boot = {num_slots = 32768 : i64}, keys = [], pts = [], results = [{{.*}}]}
// CHECK: cheddar_gpu.boot ins(%[[IN_BX]], %[[IN_AX]], %[[CONSTS]], %[[EVK]] : memref<1x65536xi64>, memref<1x65536xi64>, memref<?xi8>, memref<?xi8>) outs(%[[OUT_BX]], %[[OUT_AX]] : memref<11x65536xi64>, memref<11x65536xi64>) <num_slots = 32768, in_form = eval, out_form = eval>

// CHECK: func.func @boot_sparse_1024(%[[S_OUT_BX:[^,]+]]: memref<11x65536xi64>, %[[S_OUT_AX:[^,]+]]: memref<11x65536xi64>, %[[S_IN_BX:[^,]+]]: memref<1x65536xi64>, %[[S_IN_AX:[^,]+]]: memref<1x65536xi64>, %[[S_CONSTS:[^,]+]]: memref<?xi8>, %[[S_EVK:[^)]+]]: memref<?xi8>)
// CHECK-SAME: cheddar_gpu.abi = {args = [{{.*}}], boot = {num_slots = 1024 : i64}, keys = [], pts = [], results = [{{.*}}]}
// CHECK: cheddar_gpu.boot ins(%[[S_IN_BX]], %[[S_IN_AX]], %[[S_CONSTS]], %[[S_EVK]] : memref<1x65536xi64>, memref<1x65536xi64>, memref<?xi8>, memref<?xi8>) outs(%[[S_OUT_BX]], %[[S_OUT_AX]] : memref<11x65536xi64>, memref<11x65536xi64>) <num_slots = 1024, in_form = eval, out_form = eval>

// CHECK: func.func @boot_l2_to_l8(%[[L_OUT_BX:[^,]+]]: memref<9x65536xi64>, %[[L_OUT_AX:[^,]+]]: memref<9x65536xi64>, %[[L_IN_BX:[^,]+]]: memref<3x65536xi64>, %[[L_IN_AX:[^,]+]]: memref<3x65536xi64>, %[[L_CONSTS:[^,]+]]: memref<?xi8>, %[[L_EVK:[^)]+]]: memref<?xi8>)
// CHECK-SAME: cheddar_gpu.abi = {args = [{{.*}}], boot = {num_slots = 32768 : i64}, keys = [], pts = [], results = [{{.*}}]}
// CHECK: cheddar_gpu.boot ins(%[[L_IN_BX]], %[[L_IN_AX]], %[[L_CONSTS]], %[[L_EVK]] : memref<3x65536xi64>, memref<3x65536xi64>, memref<?xi8>, memref<?xi8>) outs(%[[L_OUT_BX]], %[[L_OUT_AX]] : memref<9x65536xi64>, memref<9x65536xi64>) <num_slots = 32768, in_form = eval, out_form = eval>

// PIPELINE: gpu.module
// PIPELINE: gpu.launch_func
// PIPELINE-NOT: ckks.
// PIPELINE-NOT: cheddar_gpu.

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [1125899908022273, 1099515691009, 1099523555329, 1099525128193, 1099526176769, 1099529060353, 1099535220737, 1099536138241, 1099537580033, 1099538104321, 1099540725761, 1099540856833, 1099543085057, 36028797019488257, 36028797023420417, 36028797024206849, 36028797025124353, 36028797032202241, 36028797033644033, 36028797037576193, 36028797048324097, 36028797048586241, 36028797049896961, 36028797051863041, 36028797053698049, 36028797054222337], P = [72057594038321153, 72057594040680449, 72057594042646529, 72057594047889409, 72057594057195521, 72057594058375169, 72057594058899457], logDefaultScale = 40>} {
  func.func @boot_default_full_slot(%arg0: !ct_L0) -> !ct_L10 {
    %0 = ckks.bootstrap %arg0 : !ct_L0 -> !ct_L10
    return %0 : !ct_L10
  }

  func.func @boot_sparse_1024(%arg0: !ct_L0) -> !ct_L10 {
    %0 = ckks.bootstrap %arg0 {cheddar.num_slots = 1024 : i64} : !ct_L0 -> !ct_L10
    return %0 : !ct_L10
  }

  func.func @boot_l2_to_l8(%arg0: !ct_L2) -> !ct_L8 {
    %0 = ckks.bootstrap %arg0 : !ct_L2 -> !ct_L8
    return %0 : !ct_L8
  }
}
