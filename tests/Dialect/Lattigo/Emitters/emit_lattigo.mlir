// RUN: heir-translate %s --emit-lattigo --split-input-file | FileCheck %s

!pk = !lattigo.rlwe.public_key
!sk = !lattigo.rlwe.secret_key
!rk = !lattigo.rlwe.relinearization_key
!gk5 = !lattigo.rlwe.galois_key<galoisElement = 5>
!eval_key_set = !lattigo.rlwe.evaluation_key_set
!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext

!encryptor = !lattigo.rlwe.encryptor<publicKey = true>
!encryptor_sk = !lattigo.rlwe.encryptor<publicKey = false>
!decryptor = !lattigo.rlwe.decryptor
!key_generator = !lattigo.rlwe.key_generator

!evaluator = !lattigo.bgv.evaluator
!encoder = !lattigo.bgv.encoder
!params = !lattigo.bgv.parameter

!value = tensor<8xi32>

#paramsLiteral = #lattigo.bgv.parameters_literal<
    logN = 14,
    logQ = [56, 55, 55],
    logP = [55],
    plaintextModulus = 0x3ee0001
>

#paramsLiteral2 = #lattigo.bgv.parameters_literal<
    logN = 14,
    Q = [65537, 17, 23],
    P = [29],
    plaintextModulus = 0x3ee0001
>

module attributes {scheme.bgv} {
  // CHECK-LABEL: func compute
  // CHECK-SAME: ([[evaluator:.*]] *bgv.Evaluator, [[ct:.*]] *rlwe.Ciphertext, [[ct1:.*]] *rlwe.Ciphertext) (*rlwe.Ciphertext)
  // CHECK: [[ct2:[^, ].*]], [[err:.*]] := [[evaluator]].AddNew([[ct]], [[ct1]])
  // CHECK: [[ct3:[^, ].*]], [[err:.*]] := [[evaluator]].MulNew([[ct2]], [[ct1]])
  // CHECK: [[ct4:[^, ].*]], [[err:.*]] := [[evaluator]].RelinearizeNew([[ct3]])
  // CHECK: [[ct5:[^, ].*]] := [[ct4]].CopyNew()
  // CHECK: [[err:.*]] := [[evaluator]].Rescale([[ct4]], [[ct5]])
  // CHECK: [[ct6:[^, ].*]], [[err:.*]] := [[evaluator]].RotateColumnsNew([[ct5]], 1)
  // CHECK: return [[ct6]]
  func.func @compute(%evaluator : !evaluator, %ct1 : !ct, %ct2 : !ct) -> (!ct) {
    %added = lattigo.bgv.add_new %evaluator, %ct1, %ct2 : (!evaluator, !ct, !ct) -> !ct
    %mul = lattigo.bgv.mul_new %evaluator, %added, %ct2 : (!evaluator, !ct, !ct) -> !ct
    %relin = lattigo.bgv.relinearize_new %evaluator, %mul : (!evaluator, !ct) -> !ct
    %rescale = lattigo.bgv.rescale_new %evaluator, %relin : (!evaluator, !ct) -> !ct
    %rotate = lattigo.bgv.rotate_columns_new %evaluator, %rescale {offset = 1} : (!evaluator, !ct) -> !ct
    return %rotate : !ct
  }

  // CHECK-LABEL: func test_basic_emitter
  // CHECK: [[param:[^, ].*]], [[err:.*]] := bgv.NewParametersFromLiteral
  // CHECK: bgv.ParametersLiteral
  // CHECK: LogN
  // CHECK: LogQ
  // CHECK: LogP
  // CHECK: PlaintextModulus
  // CHECK: [[encoder:[^, ].*]] := bgv.NewEncoder([[param]])
  // CHECK: [[kgen:[^, ].*]] := rlwe.NewKeyGenerator([[param]])
  // CHECK: [[sk:[^, ].*]], [[pk:[^, ].*]] := [[kgen]].GenKeyPairNew()
  // CHECK: [[rk:[^, ].*]] := [[kgen]].GenRelinearizationKeyNew([[sk]])
  // CHECK: [[gk5:[^, ].*]] := [[kgen]].GenGaloisKeyNew(5, [[sk]])
  // CHECK: [[evalKeySet:[^, ].*]] := rlwe.NewMemEvaluationKeySet([[rk]], [[gk5]])
  // CHECK: [[enc:[^, ].*]] := rlwe.NewEncryptor([[param]], [[pk]])
  // CHECK: [[encSk:[^, ].*]] := rlwe.NewEncryptor([[param]], [[sk]])
  // CHECK: [[dec:[^, ].*]] := rlwe.NewDecryptor([[param]], [[sk]])
  // CHECK: [[eval:[^, ].*]] := bgv.NewEvaluator([[param]], [[evalKeySet]], false)
  // CHECK: [[value1:[^, ].*]] := []int64
  // CHECK: [[value2:[^, ].*]] := []int64
  // CHECK: [[pt1:[^, ].*]] := bgv.NewPlaintext([[param]], [[param]].MaxLevel())
  // CHECK: [[pt2:[^, ].*]] := bgv.NewPlaintext([[param]], [[param]].MaxLevel())
  // CHECK: [[value1Packed:[^, ].*]][i] = int64([[value1]][i % len([[value1]])])
  // CHECK: [[encoder]].Encode([[value1Packed]], [[pt1]])
  // CHECK: [[value2Packed:[^, ].*]][i] = int64([[value2]][i % len([[value2]])])
  // CHECK: [[encoder]].Encode([[value2Packed]], [[pt2]])
  // CHECK: [[ct1:[^, ].*]], [[err:.*]] := [[enc]].EncryptNew([[pt1]])
  // CHECK: [[ct2:[^, ].*]], [[err:.*]] := [[enc]].EncryptNew([[pt2]])
  // CHECK: [[res:[^, ].*]] := compute([[eval]], [[ct1]], [[ct2]])
  // CHECK: [[pt5:[^, ].*]] := [[dec]].DecryptNew([[res]])
  // CHECK: [[value3:[^, ].*]] := []int64
  // CHECK: [[encoder]].Decode([[pt5]], [[value3]])
  // CHECK: [[value3Converted:[^, ].*]][i] = int64([[value3]][i])
  // CHECK: [[value4:[^, ].*]] := [[value3Converted]]
  func.func @test_basic_emitter() -> () {
    %param = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    %encoder = lattigo.bgv.new_encoder %param : (!params) -> !encoder
    %kgen = lattigo.rlwe.new_key_generator %param : (!params) -> !key_generator
    %sk, %pk = lattigo.rlwe.gen_key_pair %kgen : (!key_generator) -> (!sk, !pk)
    %rk = lattigo.rlwe.gen_relinearization_key %kgen, %sk : (!key_generator, !sk) -> !rk
    %gk5 = lattigo.rlwe.gen_galois_key %kgen, %sk {galoisElement = 5} : (!key_generator, !sk) -> !gk5
    %eval_key_set = lattigo.rlwe.new_evaluation_key_set %rk, %gk5 : (!rk, !gk5) -> !eval_key_set
    %encryptor = lattigo.rlwe.new_encryptor %param, %pk : (!params, !pk) -> !encryptor
    %encryptor_sk = lattigo.rlwe.new_encryptor %param, %sk : (!params, !sk) -> !encryptor_sk
    %decryptor = lattigo.rlwe.new_decryptor %param, %sk : (!params, !sk) -> !decryptor

    %evaluator = lattigo.bgv.new_evaluator %param, %eval_key_set : (!params, !eval_key_set) -> !evaluator

    %value1 = arith.constant dense<[1, 2, 3, 10]> : tensor<4xi64>
    %value2 = arith.constant dense<[10, 2, 3, 1]> : tensor<4xi64>

    %pt_raw1 = lattigo.bgv.new_plaintext %param : (!params) -> !pt
    %pt_raw2 = lattigo.bgv.new_plaintext %param : (!params) -> !pt

    %pt1 = lattigo.bgv.encode %encoder, %value1, %pt_raw1 : (!encoder, tensor<4xi64>, !pt) -> !pt
    %pt2 = lattigo.bgv.encode %encoder, %value2, %pt_raw2 : (!encoder, tensor<4xi64>, !pt) -> !pt
    %ct1 = lattigo.rlwe.encrypt %encryptor, %pt1 : (!encryptor, !pt) -> !ct
    %ct2 = lattigo.rlwe.encrypt %encryptor, %pt2 : (!encryptor, !pt) -> !ct

    %mul = func.call @compute(%evaluator, %ct1, %ct2) : (!evaluator, !ct, !ct) -> !ct

    %dec = lattigo.rlwe.decrypt %decryptor, %mul : (!decryptor, !ct) -> !pt

    %result = arith.constant dense<0> : tensor<4xi64>

    %updated = lattigo.bgv.decode %encoder, %dec, %result : (!encoder, !pt, tensor<4xi64>) -> tensor<4xi64>

    return
  }
}

// -----


// CHECK-LABEL: test_constant
// CHECK: [[v1:.*]] := 1
// CHECK: [[v2:.*]] := []int64{1, 2}
// CHECK: [[v3:.*]] := []int64{2, 2, 2, 2}
module attributes {scheme.bgv} {
  func.func @test_constant() -> () {
    %int = arith.constant 1 : i32
    %ints = arith.constant dense<[1, 2]> : tensor<2xi32>
    %dense = arith.constant dense<2> : tensor<4xi32>
    return
  }
}

// -----

!rk = !lattigo.rlwe.relinearization_key
!gk = !lattigo.rlwe.galois_key<galoisElement = 5>
!ekset = !lattigo.rlwe.evaluation_key_set

// CHECK-LABEL: test_new_evaluation_key_set_no_relin_key
// CHECK: rlwe.NewMemEvaluationKeySet(nil, [[gk:[^, ].*]])
module attributes {scheme.bgv} {
  func.func @test_new_evaluation_key_set_no_relin_key(%gk : !gk) -> (!ekset) {
    %ekset = lattigo.rlwe.new_evaluation_key_set %gk : (!gk) -> !ekset
    return %ekset : !ekset
  }
}

// -----

!params = !lattigo.bgv.parameter
!evaluator = !lattigo.bgv.evaluator

// CHECK-LABEL: test_new_evaluator_no_key_set
// CHECK: bgv.NewEvaluator([[params:[^, ].*]], nil, false)
module attributes {scheme.bgv} {
  func.func @test_new_evaluator_no_key_set(%params : !params) -> (!evaluator) {
    %evaluator = lattigo.bgv.new_evaluator %params : (!params) -> !evaluator
    return %evaluator : !evaluator
  }
}

// -----

// CHECK-LABEL: func dot_product
// CHECK: ["bound"] = "50"
// CHECK: ["complex"] = "{test = 1.200000e+00 : f64}"
// CHECK: ["random"] = "3 : i64"
// CHECK: ["secret.secret"] = "unit"
// CHECK: ["asm.is_block_arg"] = "1"
// CHECK: ["asm.result_ssa_format"]

module attributes {scheme.bgv} {
  func.func private @__heir_debug_0(!lattigo.bgv.evaluator, !lattigo.bgv.parameter, !lattigo.bgv.encoder, !lattigo.rlwe.decryptor, !lattigo.rlwe.ciphertext)
  func.func @dot_product(%evaluator: !lattigo.bgv.evaluator, %param: !lattigo.bgv.parameter, %encoder: !lattigo.bgv.encoder, %decryptor: !lattigo.rlwe.decryptor, %ct: !lattigo.rlwe.ciphertext, %ct_0: !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext attributes {mgmt.openfhe_params = #mgmt.openfhe_params<evalAddCount = 8, keySwitchCount = 15>} {
    call @__heir_debug_0(%evaluator, %param, %encoder, %decryptor, %ct) {bound = "50", random = 3, complex = {test = 1.2}, secret.secret} : (!lattigo.bgv.evaluator, !lattigo.bgv.parameter, !lattigo.bgv.encoder, !lattigo.rlwe.decryptor, !lattigo.rlwe.ciphertext) -> ()
    return %ct : !lattigo.rlwe.ciphertext
  }
}
