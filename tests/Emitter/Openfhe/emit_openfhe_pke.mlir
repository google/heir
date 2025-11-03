// RUN: heir-translate %s --emit-openfhe-pke --split-input-file | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>

!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key

!pt = !lwe.lwe_plaintext<application_data = <message_type = i3>, plaintext_space = #plaintext_space>
!ct = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: CiphertextT test_basic_emitter(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[ARG1:[^,]*]],
// CHECK-SAME:    CiphertextT [[ARG2:[^,]*]],
// CHECK-SAME:    Plaintext [[ARG3:[^,]*]],
// CHECK-SAME:    EvalKeyT [[ARG4:[^)]*]]
// CHECK-SAME:  ) {
// CHECK-NEXT:      int64_t [[const:.*]] = 4;
// CHECK-NEXT:      const auto& [[v3:.*]] = [[CC]]->EvalAdd([[ARG1]], [[ARG2]]);
// CHECK-NEXT:      const auto& [[v4:.*]] = [[CC]]->EvalSub([[ARG1]], [[ARG2]]);
// CHECK-NEXT:      const auto& [[v5:.*]] = [[CC]]->EvalMult([[v3]], [[v4]]);
// CHECK-NEXT:      const auto& [[v6:.*]] = [[CC]]->EvalNegate([[v5]]);
// CHECK-NEXT:      const auto& [[v7:.*]] = [[CC]]->EvalSquare([[v6]]);
// CHECK-NEXT:      const auto& [[v8:.*]] = [[CC]]->EvalMult([[v7]], [[ARG3]]);
// CHECK-NEXT:      const auto& [[v9:.*]] = [[CC]]->EvalMult([[v7]], [[const]]);
// CHECK-NEXT:      const auto& [[v10:.*]] = [[CC]]->Relinearize([[v9]]);
// CHECK-NEXT:      const auto& [[v11:.*]] = [[CC]]->ModReduce([[v10]]);
// CHECK-NEXT:      const auto& [[v12:.*]] = [[CC]]->LevelReduce([[v11]], nullptr, 1);
// CHECK-NEXT:      const auto& [[v13:.*]] = [[CC]]->EvalRotate([[v12]], 4);
// CHECK-NEXT:      std::map<uint32_t, EvalKeyT> [[v14_evalkeymap:.*]] = {{[{][{]}}0, [[ARG4]]{{[}][}]}};
// CHECK-NEXT:      const auto& [[v14:.*]] = [[CC]]->EvalAutomorphism([[v13]], 0, [[v14_evalkeymap]]);
// CHECK-NEXT:      const auto& [[v15:.*]] = [[CC]]->KeySwitch([[v14]], [[ARG4]]);
// CHECK-NEXT:      return [[v15]];
// CHECK-NEXT:  }
module attributes {scheme.bgv} {
  func.func @test_basic_emitter(%cc : !cc, %input1 : !ct, %input2 : !ct, %input3: !pt, %eval_key : !ek) -> !ct {
    %const = arith.constant 4 : i64
    %add_res = openfhe.add %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
    %sub_res = openfhe.sub %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
    %mul_res = openfhe.mul %cc, %add_res, %sub_res : (!cc, !ct, !ct) -> !ct
    %neg_res = openfhe.negate %cc, %mul_res : (!cc, !ct) -> !ct
    %square_res = openfhe.square %cc, %neg_res : (!cc, !ct) -> !ct
    %mul_plain_res = openfhe.mul_plain %cc, %square_res, %input3: (!cc, !ct, !pt) -> !ct
    %mul_const_res = openfhe.mul_const %cc, %square_res, %const : (!cc, !ct, i64) -> !ct
    %relin_res = openfhe.relin %cc, %mul_const_res : (!cc, !ct) -> !ct
    %mod_reduce_res = openfhe.mod_reduce %cc, %relin_res : (!cc, !ct) -> !ct
    %level_reduce_res = openfhe.level_reduce %cc, %mod_reduce_res : (!cc, !ct) -> !ct
    %rotate_res = openfhe.rot %cc, %level_reduce_res { index = 4 } : (!cc, !ct) -> !ct
    %automorph_res = openfhe.automorph %cc, %rotate_res, %eval_key : (!cc, !ct, !ek) -> !ct
    %key_switch_res = openfhe.key_switch %cc, %automorph_res, %eval_key : (!cc, !ct, !ek) -> !ct
    return %key_switch_res: !ct
  }
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>

!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key

!tensor_pt_ty = !lwe.lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space>
!scalar_pt_ty = !lwe.lwe_plaintext<application_data = <message_type = i16>, plaintext_space = #plaintext_space>
!tensor_ct_ty = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>
!scalar_ct_ty = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: simple_sum(
// CHECK-COUNT-6: EvalRotate
// CHECK: simple_sum__encrypt(
// CHECK: MakePackedPlaintext
// CHECK: Encrypt
// CHECK: simple_sum__decrypt(
// CHECK: PlaintextT
// CHECK: Decrypt
// CHECK: int16_t
// CHECK-SAME: [0]
module attributes {scheme.ckks} {
  func.func @simple_sum(%arg0: !openfhe.crypto_context, %arg1: !tensor_ct_ty) -> !scalar_ct_ty {
    %1 = openfhe.rot %arg0, %arg1 { index = 16 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
    %2 = openfhe.add %arg0, %arg1, %1 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
    %4 = openfhe.rot %arg0, %2 { index = 8 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
    %5 = openfhe.add %arg0, %2, %4 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
    %7 = openfhe.rot %arg0, %5 { index = 4 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
    %8 = openfhe.add %arg0, %5, %7 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
    %10 = openfhe.rot %arg0, %8 { index = 2 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
    %11 = openfhe.add %arg0, %8, %10 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
    %13 = openfhe.rot %arg0, %11 { index = 1 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
    %14 = openfhe.add %arg0, %11, %13 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_ct_ty) -> !tensor_ct_ty
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi16>
    %15 = openfhe.make_packed_plaintext %arg0, %cst : (!openfhe.crypto_context, tensor<32xi16>) -> !tensor_pt_ty
    %16 = openfhe.mul_plain %arg0, %14, %15 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_pt_ty) -> !tensor_ct_ty
    %18 = openfhe.rot %arg0, %16 { index = 31 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
    %19 = lwe.reinterpret_application_data %18 : !tensor_ct_ty to !scalar_ct_ty
    return %19 : !scalar_ct_ty
  }
  func.func @simple_sum__encrypt(%arg0: !openfhe.crypto_context, %arg1: tensor<32xi16>, %arg2: !openfhe.public_key) -> !tensor_ct_ty {
    %0 = openfhe.make_packed_plaintext %arg0, %arg1 : (!openfhe.crypto_context, tensor<32xi16>) -> !tensor_pt_ty
    %1 = openfhe.encrypt %arg0, %0, %arg2 : (!openfhe.crypto_context, !tensor_pt_ty, !openfhe.public_key) -> !tensor_ct_ty
    return %1 : !tensor_ct_ty
  }
  func.func @simple_sum__decrypt(%arg0: !openfhe.crypto_context, %arg1: !scalar_ct_ty, %arg2: !openfhe.private_key) -> i16 {
    %0 = openfhe.decrypt %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !scalar_ct_ty, !openfhe.private_key) -> !scalar_pt_ty
    %1 = lwe.rlwe_decode %0 {encoding = #full_crt_packing_encoding, ring = #ring_Z65537_i64_1_x32_} : !scalar_pt_ty -> i16
    return %1 : i16
  }
  // CHECK: CiphertextT test_sub_plain(
  // CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
  // CHECK-SAME:    Plaintext [[ARG1:[^,]*]],
  // CHECK-SAME:    CiphertextT [[ARG2:[^,]*]]) {
  // CHECK-NEXT:      const auto& [[v0:.*]] = [[CC]]->EvalSub([[ARG2]], [[ARG1]]);
  // CHECK-NEXT:      return [[v0]];
  // CHECK-NEXT:  }
  func.func @test_sub_plain(%cc: !openfhe.crypto_context, %pt :!tensor_pt_ty, %ct : !tensor_ct_ty) -> !tensor_ct_ty {
    %0 = openfhe.sub_plain  %cc, %ct, %pt: (!openfhe.crypto_context, !tensor_ct_ty, !tensor_pt_ty) -> !tensor_ct_ty
    return %0 : !tensor_ct_ty
  }
}

// -----

// CHECK: test_constant
// CHECK-NEXT:  std::vector<float> [[splat:.*]](2, 1.5);
// CHECK-NEXT:  std::vector<int32_t> [[ints:.*]] = {1, 2};
// CHECK-NEXT:  std::vector<float> [[floats:.*]] = {1.5, 2.5};
// CHECK-NEXT:  std::vector<double> [[double1:.*]](16, -0.38478666543960571);
// CHECK-NEXT:  std::vector<double> [[double2:.*]](16, -1.1268185335211456E-4);
// CHECK-NEXT:  std::vector<double> [[multidim:.*]] = {1.5, 2.5};
// CHECK-NEXT:  return [[splat]];
module attributes {scheme.bgv} {
  func.func @test_constant() -> tensor<2xf32> {
    %splat = arith.constant dense<1.5> : tensor<2xf32>
    %ints = arith.constant dense<[1, 2]> : tensor<2xi32>
    %floats = arith.constant dense<[1.5, 2.5]> : tensor<2xf32>
    %cst_175 = arith.constant dense<-0.38478666543960571> : tensor<16xf64>
    %cst_176 = arith.constant dense<-1.1268185335211456E-4> : tensor<16xf64>
    %cst_2d = arith.constant dense<[[1.5, 2.5]]> : tensor<1x2xf64>
    return %splat : tensor<2xf32>
  }
}

// -----

// CHECK: test_ckks_no_plaintext_modulus
// CHECK-NOT: SetPlaintextModulus
module attributes {scheme.ckks} {
  func.func @test_ckks_no_plaintext_modulus() -> !openfhe.crypto_context {
    %0 = openfhe.gen_params  {insecure = false, mulDepth = 2 : i64, plainMod = 0 : i64, evalAddCount = 0 : i64, keySwitchCount = 0 : i64, encryptionTechniqueExtended = false} : () -> !openfhe.cc_params
    %1 = openfhe.gen_context %0 {supportFHE = false} : (!openfhe.cc_params) -> !openfhe.crypto_context
    return %1 : !openfhe.crypto_context
  }
}

// -----

!Z2147565569_i64_ = !mod_arith.int<2147565569 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L0_C0_ = #lwe.modulus_chain<elements = <2147565569 : i64>, current = 0>
!rns_L0_ = !rns.rns<!Z2147565569_i64_>
#ring_Z65537_i64_1_x8_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**8>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x8_, encoding = #full_crt_packing_encoding>
#ring_rns_L0_1_x8_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**8>>
!pt = !lwe.lwe_plaintext<application_data = <message_type = i16>, plaintext_space = #plaintext_space>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x8_, encryption_type = lsb>
!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = i16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L0_C0_>

// CHECK: __heir_debug(CryptoContextT, PrivateKeyT, CiphertextT, const std::map<std::string, std::string>&)
// CHECK: ["bound"] = "50"
// CHECK: ["complex"] = "{test = 1.200000e+00 : f64}"
// CHECK: ["random"] = "3 : i64"
// CHECK: ["secret.secret"] = "unit"
// CHECK: ["asm.is_block_arg"] = "1"
// CHECK: ["asm.result_ssa_format"]

module attributes {scheme.bgv} {
  func.func private @__heir_debug_0(!openfhe.crypto_context, !openfhe.private_key, !ct_L0_)
  func.func @add(%cc: !openfhe.crypto_context, %sk: !openfhe.private_key, %ct: !ct_L0_) -> !ct_L0_ {
    call @__heir_debug_0(%cc, %sk, %ct) {bound = "50", random = 3, complex = {test = 1.2}, secret.secret} : (!openfhe.crypto_context, !openfhe.private_key, !ct_L0_) -> ()
    return %ct : !ct_L0_
  }
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>

!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>

#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>

#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>

#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>

!ct_L0_ = !lwe.lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

module attributes {scheme.ckks} {
  // CHECK: test_func_call
  // CHECK: const auto& [[v0:.*]] = callee_secret
  func.func private @callee_secret(!openfhe.crypto_context, !ct_L0_) -> !ct_L0_
  func.func @test_func_call(%cc: !openfhe.crypto_context, %arg0: !ct_L0_) -> !ct_L0_ {
    %1 = call @callee_secret(%cc, %arg0) : (!openfhe.crypto_context, !ct_L0_) -> !ct_L0_
    return %1 : !ct_L0_
  }
}


// -----

module attributes {scheme.bgv} {
  // CHECK: test_gen_params_op
  func.func @test_gen_params_op() -> !openfhe.cc_params {
    // CHECK: CCParamsT [[PARAMS:.*]];
    // CHECK: [[PARAMS]].SetMultiplicativeDepth(2);
    // CHECK: [[PARAMS]].SetPlaintextModulus(17);
    // CHECK: [[PARAMS]].SetRingDim(16384);
    // CHECK: [[PARAMS]].SetBatchSize(8);
    // CHECK: [[PARAMS]].SetFirstModSize(59);
    // CHECK: [[PARAMS]].SetScalingModSize(59);
    // CHECK: [[PARAMS]].SetEvalAddCount(2);
    // CHECK: [[PARAMS]].SetKeySwitchCount(1);
    // CHECK: [[PARAMS]].SetDigitSize(16);
    // CHECK: [[PARAMS]].SetNumLargeDigits(2);
    // CHECK: [[PARAMS]].SetMaxRelinSkDeg(3);
    // CHECK: [[PARAMS]].SetSecurityLevel(lbcrypto::HEStd_NotSet);
    // CHECK: [[PARAMS]].SetEncryptionTechnique(EXTENDED);
    // CHECK: [[PARAMS]].SetKeySwitchTechnique(BV);
    // CHECK: [[PARAMS]].SetScalingTechnique(FIXEDMANUAL);
    %0 = openfhe.gen_params  {mulDepth = 2 : i64, plainMod = 17 : i64, ringDim = 16384, batchSize = 8, firstModSize = 59, scalingModSize = 59, evalAddCount = 2 : i64, keySwitchCount = 1 : i64, digitSize = 16, numLargeDigits = 2, maxRelinSkDeg = 3, insecure = true, encryptionTechniqueExtended = true, keySwitchingTechniqueBV = true, scalingTechniqueFixedManual = true} : () -> !openfhe.cc_params
    return %0 : !openfhe.cc_params
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: test_cleartext_binops
  func.func @test_cleartext_binops() -> i64 {
    // CHECK: int64_t [[c0:[^ ]*]] = 0;
    // CHECK: int64_t [[c1:[^ ]*]] = 1;
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    // CHECK: int64_t [[v2:[^ ]*]] = [[c0]] + [[c1]];
    // CHECK: int64_t [[v3:[^ ]*]] = [[c0]] % [[c1]];
    // CHECK: bool [[v4:[^ ]*]] = [[c0]] >= [[c1]];
    // CHECK: int64_t [[v5:[^ ]*]] = [[v4]] ? [[c0]] : [[c1]];
    %0 = arith.addi %c0, %c1 : i64
    %1 = arith.remsi %c0, %c1 : i64
    %2 = arith.cmpi sge, %c0, %c1 : i64
    %3 = arith.select %2, %c0, %c1 : i64
    return %3 : i64
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: test_concat_same
  func.func @test_concat_same() -> tensor<64xi16> {
    // CHECK: std::vector<int16_t> [[v0:.*]] =
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi16>
    // CHECK: std::vector<int16_t> [[v1:.*]];
    // CHECK: for (int i = 0; i < 2; ++i) {
    // CHECK:   v1.insert([[v1]].end(), [[v0]].begin(), [[v0]].end());
    // CHECK: }
    %v = tensor.concat dim(0) %cst, %cst : (tensor<32xi16>, tensor<32xi16>) -> tensor<64xi16>

    return %v : tensor<64xi16>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: test_concat
  func.func @test_concat() -> tensor<64xi16> {
    // CHECK: std::vector<int16_t> [[c0:.*]] =
    // CHECK: std::vector<int16_t> [[c1:.*]] =
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi16>
    %cst0 = arith.constant dense<[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<32xi16>
    // CHECK: std::vector<int16_t> [[v1:.*]];
    // CHECK: [[v1]].insert([[v1]].end(), [[c0]].begin(), [[c0]].end());
    // CHECK: [[v1]].insert([[v1]].end(), [[c1]].begin(), [[c1]].end());
    %v = tensor.concat dim(0) %cst, %cst0 : (tensor<32xi16>, tensor<32xi16>) -> tensor<64xi16>

    return %v : tensor<64xi16>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: test_concat_multidim
  func.func @test_concat_multidim() -> tensor<4x16xi16> {
    // CHECK: std::vector<int16_t> [[c0:.*]] =
    // CHECK: std::vector<int16_t> [[c1:.*]] =
    %cst = arith.constant dense<[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]> : tensor<2x16xi16>
    %cst0 = arith.constant dense<[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]> : tensor<2x16xi16>
    // CHECK: std::vector<int16_t> [[v1:.*]];
    // CHECK: [[v1]].insert([[v1]].end(), [[c0]].begin(), [[c0]].end());
    // CHECK: [[v1]].insert([[v1]].end(), [[c1]].begin(), [[c1]].end());
    %v = tensor.concat dim(0) %cst, %cst0 : (tensor<2x16xi16>, tensor<2x16xi16>) -> tensor<4x16xi16>
    return %v : tensor<4x16xi16>
  }
}

// -----

module attributes {scheme.ckks} {
  // CHECK: test_insert_slice_1d
  // CHECK: std::vector<float> [[v4:[^(]*]](8, 0.100000001);
  // CHECK: std::vector<float> [[v6:[^(]*]](32);
  // CHECK: std::vector<float> [[v7:[^(]*]]([[v6]]);
  // CHECK: std::copy([[v4]].begin(), [[v4]].end(), [[v7]].begin() + 0);
  // CHECK: std::vector<float> [[v8:[^(]*]]([[v7]]);
  // CHECK: std::copy([[v4]].begin(), [[v4]].end(), [[v8]].begin() + 8);
  // CHECK: std::vector<float> [[v9:[^(]*]]([[v8]]);
  // CHECK: std::copy([[v4]].begin(), [[v4]].end(), [[v9]].begin() + 16);
  // CHECK: std::vector<float> [[v10:[^(]*]]([[v9]]);
  // CHECK: std::copy([[v4]].begin(), [[v4]].end(), [[v10]].begin() + 24);
  func.func @test_insert_slice_1d() -> tensor<32xf32> {
    %cst_2 = arith.constant dense<1.000000e-01> : tensor<8xf32>
    %0 = tensor.empty() : tensor<32xf32>
    %inserted_slice = tensor.insert_slice %cst_2 into %0[0] [8] [1] : tensor<8xf32> into tensor<32xf32>
    %inserted_slice_3 = tensor.insert_slice %cst_2 into %inserted_slice[8] [8] [1] : tensor<8xf32> into tensor<32xf32>
    %inserted_slice_4 = tensor.insert_slice %cst_2 into %inserted_slice_3[16] [8] [1] : tensor<8xf32> into tensor<32xf32>
    %inserted_slice_5 = tensor.insert_slice %cst_2 into %inserted_slice_4[24] [8] [1] : tensor<8xf32> into tensor<32xf32>
    return %inserted_slice_5 : tensor<32xf32>
  }
}

// -----

module attributes {scheme.ckks} {
  // CHECK: test_insert_slice_2d
  // CHECK: std::vector<float> [[v0:[^(]*]](64, 0.100000001);
  // CHECK: std::vector<float> [[v1:[^(]*]](1024);
  // CHECK: std::vector<float> [[v2:[^(]*]]([[v1]]);

  // TODO(#1703): this test is quite wrong, but only because the type
  // declaration and initializations above are bad, which is owned by a
  // different part of the emitter, while the loop itself is good.

  // CHECK:  int64_t [[v0]]_0 = 0;
  // CHECK:  for (int64_t [[v1]]_0 = 8; [[v1]]_0 < 24; [[v1]]_0 += 2) {
  // CHECK:    int64_t [[v0]]_1 = 0;
  // CHECK:    for (int64_t [[v1]]_1 = 8; [[v1]]_1 < 24; [[v1]]_1 += 2) {
  // CHECK:      [[v2]]{{\[}}[[v1]]_1 + 32 * ([[v1]]_0)] = [[v0]]{{\[}}[[v0]]_1 + 8 * ([[v0]]_0)];
  // CHECK:      [[v0]]_1 += 1;
  // CHECK:    }
  // CHECK:    [[v0]]_0 += 1;
  // CHECK:  }
  func.func @test_insert_slice_2d() -> tensor<32x32xf32> {
    %cst_2 = arith.constant dense<1.000000e-01> : tensor<8x8xf32>
    %0 = tensor.empty() : tensor<32x32xf32>
    %inserted_slice = tensor.insert_slice %cst_2 into %0[8, 8] [8, 8] [2, 2] : tensor<8x8xf32> into tensor<32x32xf32>
    return %inserted_slice : tensor<32x32xf32>
  }
}

// -----

module attributes {scheme.ckks} {
  // CHECK: test_extract_slice_1d
  // CHECK: std::vector<float> [[source:[^(]*]](32, 0.100000001);
  // CHECK: std::vector<float> [[result:[^(]*]](8);
  // CHECK: std::copy([[source]].begin() + 8, [[source]].begin() + 8 + 8, [[result]].begin());
  func.func @test_extract_slice_1d() -> tensor<8xf32> {
    %cst = arith.constant dense<1.000000e-01> : tensor<32xf32>
    %result = tensor.extract_slice %cst[8] [8] [1] : tensor<32xf32> to tensor<8xf32>
    return %result : tensor<8xf32>
  }
}

// -----

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
!cc = !openfhe.crypto_context
!ct = !lwe.lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: CiphertextT test_fast_rot(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[ARG1:[^,]*]]) {
// CHECK-NEXT:      const auto& [[v3:.*]] = [[CC]]->EvalFastRotationPrecompute([[ARG1]]);
// CHECK-NEXT:      const auto& [[v4:.*]] = [[CC]]->EvalFastRotation([[ARG1]], 4, 2 * [[CC]]->GetRingDimension(), [[v3]]);
// CHECK-NEXT:      return [[v4]];
// CHECK-NEXT:  }
module attributes {scheme.ckks} {
  func.func @test_fast_rot(%cc: !cc, %input1: !ct) -> !ct {
    %precomp = openfhe.fast_rotation_precompute %cc, %input1 : (!cc, !ct) -> !openfhe.digit_decomp
    %res = openfhe.fast_rotation %cc, %input1, %precomp {index = 4 : index, cyclotomicOrder = 64 : index} : (!cc, !ct, !openfhe.digit_decomp) -> !ct
    return %res : !ct
  }
}
