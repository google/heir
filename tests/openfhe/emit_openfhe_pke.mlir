// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start=30, cleartext_bitwidth=3>

#my_poly = #polynomial.polynomial<1 + x**16384>
#ring= #polynomial.ring<cmod=7917, ideal=#my_poly>
#params = #lwe.rlwe_params<dimension=1, ring=#ring>
!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key
!pt = !lwe.rlwe_plaintext<encoding = #encoding>
!ct = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params>

// CHECK-LABEL: CiphertextT test_basic_emitter(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[ARG1:[^,]*]],
// CHECK-SAME:    CiphertextT [[ARG2:[^,]*]],
// CHECK-SAME:    Plaintext [[ARG3:[^,]*]],
// CHECK-SAME:    EvalKeyT [[ARG4:[^)]*]]
// CHECK-SAME:  ) {
// CHECK-NEXT:      auto [[const:.*]] = 4;
// CHECK-NEXT:      auto [[v3:.*]] = [[CC]]->EvalAdd([[ARG1]], [[ARG2]]);
// CHECK-NEXT:      auto [[v4:.*]] = [[CC]]->EvalSub([[ARG1]], [[ARG2]]);
// CHECK-NEXT:      auto [[v5:.*]] = [[CC]]->EvalMult([[v3]], [[v4]]);
// CHECK-NEXT:      auto [[v6:.*]] = [[CC]]->EvalNegate([[v5]]);
// CHECK-NEXT:      auto [[v7:.*]] = [[CC]]->EvalSquare([[v6]]);
// CHECK-NEXT:      auto [[v8:.*]] = [[CC]]->EvalMult([[v7]], [[ARG3]]);
// CHECK-NEXT:      auto [[v9:.*]] = [[CC]]->EvalMult([[v7]], [[const]]);
// CHECK-NEXT:      auto [[v10:.*]] = [[CC]]->Relinearize([[v9]]);
// CHECK-NEXT:      auto [[v11:.*]] = [[CC]]->ModReduce([[v10]]);
// CHECK-NEXT:      auto [[v12:.*]] = [[CC]]->LevelReduce([[v11]]);
// CHECK-NEXT:      auto [[v13:.*]] = [[CC]]->EvalRotate([[v12]], [[const]]);
// CHECK-NEXT:      std::map<uint32_t, EvalKeyT> [[v14_evalkeymap:.*]] = {{[{][{]}}0, [[ARG4]]{{[}][}]}};
// CHECK-NEXT:      auto [[v14:.*]] = [[CC]]->EvalAutomorphism([[v13]], 0, [[v14_evalkeymap]]);
// CHECK-NEXT:      auto [[v15:.*]] = [[CC]]->KeySwitch([[v14]], [[ARG4]]);
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
  %rotate_res = openfhe.rot %cc, %level_reduce_res, %const : (!cc, !ct, i64) -> !ct
  %automorph_res = openfhe.automorph %cc, %rotate_res, %eval_key : (!cc, !ct, !ek) -> !ct
  %key_switch_res = openfhe.key_switch %cc, %automorph_res, %eval_key : (!cc, !ct, !ek) -> !ct
  return %key_switch_res: !ct
}
