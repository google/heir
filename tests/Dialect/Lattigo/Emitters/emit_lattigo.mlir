// RUN: heir-translate %s --emit-lattigo --split-input-file | FileCheck %s

!pk = !lattigo.rlwe.public_key
!sk = !lattigo.rlwe.secret_key
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
  // CHECK: [[v3:v.*]], err := [[v0]].AddNew([[v1]], [[v2]])
  // CHECK: [[v4:v.*]], err := [[v0]].MulNew([[v3]], [[v2]])
  // CHECK: return [[v4]]
  func.func @compute(%evaluator : !lattigo.bgv.evaluator, %ct1 : !lattigo.rlwe.ciphertext, %ct2 : !lattigo.rlwe.ciphertext) -> (!lattigo.rlwe.ciphertext) {
    %added = lattigo.bgv.add %evaluator, %ct1, %ct2 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
    %mul = lattigo.bgv.mul %evaluator, %added, %ct2 : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
    return %mul : !lattigo.rlwe.ciphertext
  }

  // CHECK-LABEL: func test_basic_emitter
  // CHECK: [[v1:v.*]], err := bgv.NewParametersFromLiteral
  // CHECK: bgv.ParametersLiteral
  // CHECK: LogN
  // CHECK: LogQ
  // CHECK: LogP
  // CHECK: PlaintextModulus
  // CHECK: [[v2:v.*]] := bgv.NewEncoder([[v1]])
  // CHECK: [[v3:v.*]] := rlwe.NewKeyGenerator([[v1]])
  // CHECK: [[v4:v.*]], [[v5:v.*]] := [[v3]].GenKeyPairNew()
  // CHECK: [[v6:v.*]] := rlwe.NewEncryptor([[v1]], [[v5]])
  // CHECK: [[v7:v.*]] := rlwe.NewDecryptor([[v1]], [[v4]])
  // CHECK: [[v8:v.*]] := bgv.NewEvaluator([[v1]], nil)
  // CHECK: [[v9:v.*]] := []int64
  // CHECK: [[v10:v.*]] := []int64
  // CHECK: [[v11:v.*]] := bgv.NewPlaintext([[v1]], [[v1]].MaxLevel())
  // CHECK: [[v12:v.*]] := bgv.NewPlaintext([[v1]], [[v1]].MaxLevel())
  // CHECK: [[v2]].Encode([[v9]], [[v11]])
  // CHECK: [[v13:v.*]] := [[v11]]
  // CHECK: [[v2]].Encode([[v10]], [[v12]])
  // CHECK: [[v14:v.*]] := [[v12]]
  // CHECK: [[v15:v.*]], err := [[v6]].EncryptNew([[v13]])
  // CHECK: [[v16:v.*]], err := [[v6]].EncryptNew([[v14]])
  // CHECK: [[v17:v.*]] := compute([[v8]], [[v15]], [[v16]])
  // CHECK: [[v18:v.*]] := [[v7]].DecryptNew([[v17]])
  // CHECK: [[v19:v.*]] := []int64
  // CHECK: [[v2]].Decode([[v18]], [[v19]])
  // CHECK: [[v20:v.*]] := [[v19]]
  func.func @test_basic_emitter() -> () {
    %param = lattigo.bgv.new_parameters_from_literal {paramsLiteral = #paramsLiteral} : () -> !lattigo.bgv.parameter
    %encoder = lattigo.bgv.new_encoder %param : (!lattigo.bgv.parameter) -> !lattigo.bgv.encoder
    %kgen = lattigo.rlwe.new_key_generator %param : (!lattigo.bgv.parameter) -> !lattigo.rlwe.key_generator
    %sk, %pk = lattigo.rlwe.gen_key_pair %kgen : (!lattigo.rlwe.key_generator) -> (!lattigo.rlwe.secret_key, !lattigo.rlwe.public_key)
    %encryptor = lattigo.rlwe.new_encryptor %param, %pk : (!lattigo.bgv.parameter, !lattigo.rlwe.public_key) -> !lattigo.rlwe.encryptor
    %decryptor = lattigo.rlwe.new_decryptor %param, %sk : (!lattigo.bgv.parameter, !lattigo.rlwe.secret_key) -> !lattigo.rlwe.decryptor

    %evaluator = lattigo.bgv.new_evaluator %param : (!lattigo.bgv.parameter) -> !lattigo.bgv.evaluator

    %value1 = arith.constant dense<10> : tensor<8xi64>
    %value2 = arith.constant dense<10> : tensor<8xi64>

    %pt_raw1 = lattigo.bgv.new_plaintext %param : (!lattigo.bgv.parameter) -> !lattigo.rlwe.plaintext
    %pt_raw2 = lattigo.bgv.new_plaintext %param : (!lattigo.bgv.parameter) -> !lattigo.rlwe.plaintext

    %pt1 = lattigo.bgv.encode %encoder, %value1, %pt_raw1 : (!lattigo.bgv.encoder, tensor<8xi64>, !lattigo.rlwe.plaintext) -> !lattigo.rlwe.plaintext
    %pt2 = lattigo.bgv.encode %encoder, %value2, %pt_raw2 : (!lattigo.bgv.encoder, tensor<8xi64>, !lattigo.rlwe.plaintext) -> !lattigo.rlwe.plaintext
    %ct1 = lattigo.rlwe.encrypt %encryptor, %pt1 : (!lattigo.rlwe.encryptor, !lattigo.rlwe.plaintext) -> !lattigo.rlwe.ciphertext
    %ct2 = lattigo.rlwe.encrypt %encryptor, %pt2 : (!lattigo.rlwe.encryptor, !lattigo.rlwe.plaintext) -> !lattigo.rlwe.ciphertext

    %mul = func.call @compute(%evaluator, %ct1, %ct2) : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext

    %dec = lattigo.rlwe.decrypt %decryptor, %mul : (!lattigo.rlwe.decryptor, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.plaintext

    %result = arith.constant dense<0> : tensor<8xi64>

    %updated = lattigo.bgv.decode %encoder, %dec, %result : (!lattigo.bgv.encoder, !lattigo.rlwe.plaintext, tensor<8xi64>) -> tensor<8xi64>

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
