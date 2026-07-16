// RUN: heir-opt --convert-to-emitc %s | FileCheck %s

!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key
!pt = !openfhe.plaintext
!ct = !openfhe.ciphertext
!pk = !openfhe.public_key
!sk = !openfhe.private_key
!params = !openfhe.cc_params

module attributes {scheme.bgv} {
  // CHECK: emitc.func @test_basic_emitter
  // CHECK: %[[CONST:.*]] = "emitc.constant"() <{value = 4 : i64}> : () -> i64
  // CHECK: %[[ADD:.*]] = member_call_opaque %{{.*}} "EvalAdd"(%[[ARG1:.*]], %[[ARG2:.*]])
  // CHECK: %[[SUB:.*]] = member_call_opaque %{{.*}} "EvalSub"(%[[ARG1]], %[[ARG2]])
  // CHECK: %[[MUL:.*]] = member_call_opaque %{{.*}} "EvalMult"(%[[ADD]], %[[SUB]])
  // CHECK: %[[MULPLAIN:.*]] = member_call_opaque %{{.*}} "EvalMult"(%[[MUL]], %[[ARG3:.*]])
  // CHECK: %[[ROTCST:.*]] = "emitc.constant"() <{value = 4 : index}> : () -> !emitc.size_t
  // CHECK: %[[ROT:.*]] = member_call_opaque %{{.*}} "EvalRotate"(%[[MULPLAIN]], %[[ROTCST]])
  // CHECK: return %[[ROT]] : !emitc.opaque<"CiphertextT">
  func.func @test_basic_emitter(%cc : !cc, %input1 : !ct, %input2 : !ct, %input3: !pt, %eval_key : !ek) -> !ct {
    %const = arith.constant 4 : i64
    %add_res = openfhe.add %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
    %sub_res = openfhe.sub %cc, %input1, %input2 : (!cc, !ct, !ct) -> !ct
    %mul_res = openfhe.mul %cc, %add_res, %sub_res : (!cc, !ct, !ct) -> !ct
    %mul_plain_res = openfhe.mul_plain %cc, %mul_res, %input3: (!cc, !ct, !pt) -> !ct
    %rotate_res = openfhe.rot %cc, %mul_plain_res { static_shift = 4 : index } : (!cc, !ct) -> !ct
    return %rotate_res: !ct
  }

  // CHECK: emitc.func @dot_product
  // CHECK: %[[CT1:.*]] = member_call_opaque %{{.*}} "EvalMultNoRelin"(%[[CT:.*]], %[[CT0:.*]])
  // CHECK: member_call_opaque %{{.*}} "RelinearizeInPlace"(%[[CT1]])
  // CHECK: %[[ROTCST4:.*]] = "emitc.constant"() <{value = 4 : index}> : () -> !emitc.size_t
  // CHECK: %[[CT3:.*]] = member_call_opaque %{{.*}} "EvalRotate"(%[[CT1]], %[[ROTCST4]])
  // CHECK: member_call_opaque %{{.*}} "EvalAddInPlace"(%[[CT1]], %[[CT3]])
  // CHECK: %[[ROTCST2:.*]] = "emitc.constant"() <{value = 2 : index}> : () -> !emitc.size_t
  // CHECK: %[[CT5:.*]] = member_call_opaque %{{.*}} "EvalRotate"(%[[CT1]], %[[ROTCST2]])
  // CHECK: member_call_opaque %{{.*}} "EvalAddInPlace"(%[[CT1]], %[[CT5]])
  // CHECK: %[[ROTCST1:.*]] = "emitc.constant"() <{value = 1 : index}> : () -> !emitc.size_t
  // CHECK: %[[CT7:.*]] = member_call_opaque %{{.*}} "EvalRotate"(%[[CT1]], %[[ROTCST1]])
  // CHECK: member_call_opaque %{{.*}} "EvalAddInPlace"(%[[CT1]], %[[CT7]])
  // CHECK: member_call_opaque %{{.*}} "ModReduceInPlace"(%[[CT1]])
  // CHECK: %[[PT:.*]] = member_call_opaque %{{.*}} "MakePackedPlaintext"(%{{.*}})
  // CHECK: %[[CT10:.*]] = member_call_opaque %{{.*}} "EvalMult"(%[[CT1]], %[[PT]])
  // CHECK: %[[ROTCST7:.*]] = "emitc.constant"() <{value = 7 : index}> : () -> !emitc.size_t
  // CHECK: %[[CT11:.*]] = member_call_opaque %{{.*}} "EvalRotate"(%[[CT10]], %[[ROTCST7]])
  // CHECK: member_call_opaque %{{.*}} "ModReduceInPlace"(%[[CT11]])
  // CHECK: return %[[CT11]] : !emitc.opaque<"CiphertextT">
  func.func @dot_product(%cc: !cc, %ct: !ct, %ct_0: !ct) -> !ct {
    %ct_1 = openfhe.mul_no_relin %cc, %ct, %ct_0 : (!cc, !ct, !ct) -> !ct
    %ct_2 = openfhe.relin_inplace %cc, %ct_1 : (!cc, !ct) -> !ct
    %ct_3 = openfhe.rot %cc, %ct_2 {static_shift = 4 : index} : (!cc, !ct) -> !ct
    %ct_4 = openfhe.add_inplace %cc, %ct_2, %ct_3 : (!cc, !ct, !ct) -> !ct
    %ct_5 = openfhe.rot %cc, %ct_4 {static_shift = 2 : index} : (!cc, !ct) -> !ct
    %ct_6 = openfhe.add_inplace %cc, %ct_4, %ct_5 : (!cc, !ct, !ct) -> !ct
    %ct_7 = openfhe.rot %cc, %ct_6 {static_shift = 1 : index} : (!cc, !ct) -> !ct
    %ct_8 = openfhe.add_inplace %cc, %ct_6, %ct_7 : (!cc, !ct, !ct) -> !ct
    %ct_9 = openfhe.mod_reduce_inplace %cc, %ct_8 : (!cc, !ct) -> !ct
    %cst = memref.alloc() : memref<8xi16>
    %pt = openfhe.make_packed_plaintext %cc, %cst : (!cc, memref<8xi16>) -> !pt
    %ct_10 = openfhe.mul_plain %cc, %ct_9, %pt : (!cc, !ct, !pt) -> !ct
    %ct_11 = openfhe.rot %cc, %ct_10 {static_shift = 7 : index} : (!cc, !ct) -> !ct
    %ct_12 = openfhe.mod_reduce_inplace %cc, %ct_11 : (!cc, !ct) -> !ct
    return %ct_12 : !ct
  }

  // CHECK: emitc.func @test_sub_inplace
  // CHECK-SAME: (%[[CC:.*]]: !emitc.opaque<"CryptoContextT">, %[[CT:.*]]: !emitc.opaque<"CiphertextT">) -> !emitc.opaque<"CiphertextT">
  func.func @test_sub_inplace(%cc: !cc, %ct: !ct) -> !ct {
    // CHECK: member_call_opaque %{{.*}} "EvalSubInPlace"(%[[CT]], %[[CT]])
    %0 = openfhe.sub_inplace %cc, %ct, %ct : (!cc, !ct, !ct) -> !ct
    // CHECK: return %[[CT]]
    return %0 : !ct
  }

  // CHECK: emitc.func @test_setup
  // CHECK-SAME: (%[[CC:.*]]: !emitc.opaque<"CryptoContextT">, %[[SK:.*]]: !emitc.opaque<"PrivateKeyT">)
  func.func @test_setup(%cc: !cc, %sk: !sk) {
    // CHECK: %[[PARAMS:.*]] = call_opaque "CCParamsT"()
    // CHECK: member_call_opaque %[[PARAMS]] "SetMultiplicativeDepth"() <{args = [1]}>
    // CHECK: member_call_opaque %[[PARAMS]] "SetPlaintextModulus"() <{args = [2]}>
    // CHECK: member_call_opaque %[[PARAMS]] "SetKeySwitchTechnique"() <{args = [#emitc.opaque<"HYBRID">]}>
    %params = openfhe.gen_params { mulDepth = 1 : i64, plainMod = 2 : i64 } : () -> !params

    // CHECK: %[[NEW_CC:.*]] = call_opaque "GenCryptoContext"(%[[PARAMS]])
    // CHECK: member_call_opaque %{{.*}} "Enable"() <{args = [#emitc.opaque<"PKE">]}>
    // CHECK: member_call_opaque %{{.*}} "Enable"() <{args = [#emitc.opaque<"KEYSWITCH">]}>
    // CHECK: member_call_opaque %{{.*}} "Enable"() <{args = [#emitc.opaque<"LEVELEDSHE">]}>
    // CHECK: member_call_opaque %{{.*}} "Enable"() <{args = [#emitc.opaque<"ADVANCEDSHE">]}>
    // CHECK: member_call_opaque %{{.*}} "Enable"() <{args = [#emitc.opaque<"FHE">]}>
    %new_cc = openfhe.gen_context %params { supportFHE = true } : (!params) -> !cc

    // CHECK: member_call_opaque %{{.*}} "EvalMultKeyGen"(%[[SK]])
    openfhe.gen_mulkey %cc, %sk : (!cc, !sk) -> ()

    // CHECK: %[[VEC_CONST:.*]] = "emitc.constant"() <{value = #emitc.opaque<"{{.*}}">}> : () -> !emitc.opaque<"std::vector<int32_t>">
    // CHECK: member_call_opaque %{{.*}} "EvalRotateKeyGen"(%[[SK]], %[[VEC_CONST]])
    openfhe.gen_rotkey %cc, %sk { indices = array<i64: 1, 2> } : (!cc, !sk) -> ()
    return
  }

  // CHECK: emitc.func @test_crypt
  // CHECK-SAME: (%[[CC:.*]]: !emitc.opaque<"CryptoContextT">, %[[PT:.*]]: !emitc.opaque<"Plaintext">, %[[PK:.*]]: !emitc.opaque<"PublicKeyT">, %[[SK:.*]]: !emitc.opaque<"PrivateKeyT">) -> !emitc.ptr<i64>
  func.func @test_crypt(%cc: !cc, %pt: !pt, %pk: !pk, %sk: !sk) -> memref<8xi64> {
    // CHECK: %[[ENC:.*]] = member_call_opaque %{{.*}} "Encrypt"(%[[PK]], %[[PT]])
    %encrypted = openfhe.encrypt %cc, %pt, %pk : (!cc, !pt, !pk) -> !ct

    // CHECK: %[[DEC_VAR:.*]] = "emitc.variable"() <{value = #emitc.opaque<"Plaintext()">}> : () -> !emitc.lvalue<!emitc.opaque<"Plaintext">>
    // CHECK: %[[DEC_ADDR:.*]] = address_of %[[DEC_VAR]] : !emitc.lvalue<!emitc.opaque<"Plaintext">>
    // CHECK: member_call_opaque %{{.*}} "Decrypt"(%[[SK]], %[[ENC]], %[[DEC_ADDR]])
    // CHECK: %[[DEC_VAL:.*]] = load %[[DEC_VAR]] : <!emitc.opaque<"Plaintext">>
    %decrypted = openfhe.decrypt %cc, %encrypted, %sk : (!cc, !ct, !sk) -> !pt

    // CHECK: %[[VEC_REF:.*]] = member_call_opaque %{{.*}} "GetPackedValue"()
    // CHECK: %[[DATA_PTR:.*]] = member_call_opaque %[[VEC_REF]] "data"()
    %decoded = openfhe.decode %decrypted : !pt -> memref<8xi64>

    // CHECK: return %[[DATA_PTR]]
    return %decoded : memref<8xi64>
  }
}
