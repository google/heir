// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.int_polynomial<1 + x**16384>
#ring= #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#params = #lwe.rlwe_params<dimension=1, ring=#ring>
!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key
!pt = !lwe.rlwe_plaintext<encoding = #encoding, ring=#ring, underlying_type=i3>
!ct = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type=i3>

// CHECK-LABEL: CiphertextT test_basic_emitter(
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
// CHECK-NEXT:      const auto& [[v12:.*]] = [[CC]]->LevelReduce([[v11]]);
// CHECK-NEXT:      const auto& [[v13:.*]] = [[CC]]->EvalRotate([[v12]], 4);
// CHECK-NEXT:      std::map<uint32_t, EvalKeyT> [[v14_evalkeymap:.*]] = {{[{][{]}}0, [[ARG4]]{{[}][}]}};
// CHECK-NEXT:      const auto& [[v14:.*]] = [[CC]]->EvalAutomorphism([[v13]], 0, [[v14_evalkeymap]]);
// CHECK-NEXT:      const auto& [[v15:.*]] = [[CC]]->KeySwitch([[v14]], [[ARG4]]);
// CHECK-NEXT:      return [[v15]];
// CHECK-NEXT:  }
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

#degree_32_poly = #polynomial.int_polynomial<1 + x**32>
#eval_encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
#ring2 = #polynomial.ring<coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#degree_32_poly>
#params2 = #lwe.rlwe_params<ring = #ring2>
!tensor_pt_ty = !lwe.rlwe_plaintext<encoding = #eval_encoding, ring = #ring2, underlying_type = tensor<32xi16>>
!scalar_pt_ty = !lwe.rlwe_plaintext<encoding = #eval_encoding, ring = #ring2, underlying_type = i16>
!tensor_ct_ty = !lwe.rlwe_ciphertext<encoding = #eval_encoding, rlwe_params = #params2, underlying_type = tensor<32xi16>>
!scalar_ct_ty = !lwe.rlwe_ciphertext<encoding = #eval_encoding, rlwe_params = #params2, underlying_type = i16>

// CHECK-LABEL: simple_sum(
// CHECK-COUNT-6: EvalRotate
// CHECK-LABEL: simple_sum__encrypt(
// CHECK: MakePackedPlaintext
// CHECK: Encrypt
// CHECK-LABEL: simple_sum__decrypt(
// CHECK: PlaintextT
// CHECK: Decrypt
// CHECK: int16_t
// CHECK-SAME: [0]
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
  %15 = lwe.rlwe_encode %cst {encoding = #eval_encoding, ring = #ring2} : tensor<32xi16> -> !tensor_pt_ty
  %16 = openfhe.mul_plain %arg0, %14, %15 : (!openfhe.crypto_context, !tensor_ct_ty, !tensor_pt_ty) -> !tensor_ct_ty
  %18 = openfhe.rot %arg0, %16 { index = 31 } : (!openfhe.crypto_context, !tensor_ct_ty) -> !tensor_ct_ty
  %19 = lwe.reinterpret_underlying_type %18 : !tensor_ct_ty to !scalar_ct_ty
  return %19 : !scalar_ct_ty
}
func.func @simple_sum__encrypt(%arg0: !openfhe.crypto_context, %arg1: tensor<32xi16>, %arg2: !openfhe.public_key) -> !tensor_ct_ty {
  %0 = lwe.rlwe_encode %arg1 {encoding = #eval_encoding, ring = #ring2} : tensor<32xi16> -> !tensor_pt_ty
  %1 = openfhe.encrypt %arg0, %0, %arg2 : (!openfhe.crypto_context, !tensor_pt_ty, !openfhe.public_key) -> !tensor_ct_ty
  return %1 : !tensor_ct_ty
}
func.func @simple_sum__decrypt(%arg0: !openfhe.crypto_context, %arg1: !scalar_ct_ty, %arg2: !openfhe.private_key) -> i16 {
  %0 = openfhe.decrypt %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !scalar_ct_ty, !openfhe.private_key) -> !scalar_pt_ty
  %1 = lwe.rlwe_decode %0 {encoding = #eval_encoding, ring = #ring2} : !scalar_pt_ty -> i16
  return %1 : i16
}
