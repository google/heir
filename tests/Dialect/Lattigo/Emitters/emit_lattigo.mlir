// RUN: heir-translate %s --emit-lattigo --split-input-file | FileCheck %s

!pk = !lattigo.rlwe.public_key
!sk = !lattigo.rlwe.secret_key
!rk = !lattigo.rlwe.relinearization_key
!gk5 = !lattigo.rlwe.galois_key<galoisElement = 5>
!eval_key_set = !lattigo.rlwe.evaluation_key_set
!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext

!encryptor = !lattigo.rlwe.encryptor
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

module {
  // CHECK-LABEL: func compute
  // CHECK-SAME: ([[v0:v.*]] *bgv.Evaluator, [[v1:v.*]] *rlwe.Ciphertext, [[v2:v.*]] *rlwe.Ciphertext) (*rlwe.Ciphertext)
  // CHECK: [[v3:v.*]], [[err:.*]] := [[v0]].AddNew([[v1]], [[v2]])
  // CHECK: [[v4:v.*]], [[err:.*]] := [[v0]].MulNew([[v3]], [[v2]])
  // CHECK: [[v5:v.*]], [[err:.*]] := [[v0]].RelinearizeNew([[v4]])
  // CHECK: [[err:.*]] := [[v0]].Rescale([[v5]], [[v5]])
  // CHECK: [[v6:v.*]] := [[v5]]
  // CHECK: [[v7:v.*]], [[err:.*]] := [[v0]].RotateColumnsNew([[v6]], 1)
  // CHECK: return [[v7]]
  func.func @compute(%evaluator : !evaluator, %ct1 : !ct, %ct2 : !ct) -> (!ct) {
    %added = lattigo.bgv.add %evaluator, %ct1, %ct2 : (!evaluator, !ct, !ct) -> !ct
    %mul = lattigo.bgv.mul %evaluator, %added, %ct2 : (!evaluator, !ct, !ct) -> !ct
    %relin = lattigo.bgv.relinearize %evaluator, %mul : (!evaluator, !ct) -> !ct
    %rescale = lattigo.bgv.rescale %evaluator, %relin : (!evaluator, !ct) -> !ct
    %rotate = lattigo.bgv.rotate_columns %evaluator, %rescale {offset = 1} : (!evaluator, !ct) -> !ct
    return %rotate : !ct
  }

  // CHECK-LABEL: func test_basic_emitter
  // CHECK: [[param:v.*]], [[err:.*]] := bgv.NewParametersFromLiteral
  // CHECK: bgv.ParametersLiteral
  // CHECK: LogN
  // CHECK: LogQ
  // CHECK: LogP
  // CHECK: PlaintextModulus
  // CHECK: [[encoder:v.*]] := bgv.NewEncoder([[param]])
  // CHECK: [[kgen:v.*]] := rlwe.NewKeyGenerator([[param]])
  // CHECK: [[sk:v.*]], [[pk:v.*]] := [[kgen]].GenKeyPairNew()
  // CHECK: [[rk:v.*]] := [[kgen]].GenRelinearizationKeyNew([[sk]])
  // CHECK: [[gk5:v.*]] := [[kgen]].GenGaloisKeyNew(5, [[sk]])
  // CHECK: [[evalKeySet:v.*]] := rlwe.NewMemEvaluationKeySet([[rk]], [[gk5]])
  // CHECK: [[enc:v.*]] := rlwe.NewEncryptor([[param]], [[pk]])
  // CHECK: [[dec:v.*]] := rlwe.NewDecryptor([[param]], [[sk]])
  // CHECK: [[eval:v.*]] := bgv.NewEvaluator([[param]], [[evalKeySet]])
  // CHECK: [[value1:v.*]] := []int64
  // CHECK: [[value2:v.*]] := []int64
  // CHECK: [[pt1:v.*]] := bgv.NewPlaintext([[param]], [[param]].MaxLevel())
  // CHECK: [[pt2:v.*]] := bgv.NewPlaintext([[param]], [[param]].MaxLevel())
  // CHECK: [[encoder]].Encode([[value1]], [[pt1]])
  // CHECK: [[pt3:v.*]] := [[pt1]]
  // CHECK: [[encoder]].Encode([[value2]], [[pt2]])
  // CHECK: [[pt4:v.*]] := [[pt2]]
  // CHECK: [[ct1:v.*]], [[err:.*]] := [[enc]].EncryptNew([[pt3]])
  // CHECK: [[ct2:v.*]], [[err:.*]] := [[enc]].EncryptNew([[pt4]])
  // CHECK: [[res:v.*]] := compute([[eval]], [[ct1]], [[ct2]])
  // CHECK: [[pt5:v.*]] := [[dec]].DecryptNew([[res]])
  // CHECK: [[value3:v.*]] := []int64
  // CHECK: [[encoder]].Decode([[pt5]], [[value3]])
  // CHECK: [[value4:v.*]] := [[value3]]
  func.func @test_basic_emitter() -> () {
    %param = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !params
    %encoder = lattigo.bgv.new_encoder %param : (!params) -> !encoder
    %kgen = lattigo.rlwe.new_key_generator %param : (!params) -> !key_generator
    %sk, %pk = lattigo.rlwe.gen_key_pair %kgen : (!key_generator) -> (!sk, !pk)
    %rk = lattigo.rlwe.gen_relinearization_key %kgen, %sk : (!key_generator, !sk) -> !rk
    %gk5 = lattigo.rlwe.gen_galois_key %kgen, %sk {galoisElement = 5} : (!key_generator, !sk) -> !gk5
    %eval_key_set = lattigo.rlwe.new_evaluation_key_set %rk, %gk5 : (!rk, !gk5) -> !eval_key_set
    %encryptor = lattigo.rlwe.new_encryptor %param, %pk : (!params, !pk) -> !encryptor
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
// CHECK: [[v1:v.*]] := 1
// CHECK: [[v2:v.*]] := []int64{1, 2}
// CHECK: [[v3:v.*]] := []int64{2, 2, 2, 2}
func.func @test_constant() -> () {
  %int = arith.constant 1 : i32
  %ints = arith.constant dense<[1, 2]> : tensor<2xi32>
  %dense = arith.constant dense<2> : tensor<4xi32>
  return
}
