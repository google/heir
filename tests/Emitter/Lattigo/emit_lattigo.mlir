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
  // CHECK: func compute
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

  // CHECK: func test_basic_emitter
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
  // CHECK: [[value1:[^, ].*]] := []int32
  // CHECK: [[value2:[^, ].*]] := []int32
  // CHECK: [[pt1:[^, ].*]] := bgv.NewPlaintext([[param]], [[param]].MaxLevel())
  // CHECK: [[pt2:[^, ].*]] := bgv.NewPlaintext([[param]], [[param]].MaxLevel())
  // CHECK: [[value1Packed:[^, ].*]][i] = int64([[value1]][i])
  // CHECK: [[encoder]].Encode([[value1Packed]], [[pt1]])
  // CHECK: [[value2Packed:[^, ].*]][i] = int64([[value2]][i])
  // CHECK: [[encoder]].Encode([[value2Packed]], [[pt2]])
  // CHECK: [[ct1:[^, ].*]], [[err:.*]] := [[enc]].EncryptNew([[pt1]])
  // CHECK: [[ct2:[^, ].*]], [[err:.*]] := [[enc]].EncryptNew([[pt2]])
  // CHECK: [[res:[^, ].*]] := compute([[eval]], [[ct1]], [[ct2]])
  // CHECK: [[pt5:[^, ].*]] := [[dec]].DecryptNew([[res]])
  // CHECK: [[value3:[^, ].*]] := []int32
  // CHECK: [[value3Int64:[^, ].*]] := make([]int64, len([[value3]]))
  // CHECK: [[encoder]].Decode([[pt5]], [[value3Int64]])
  // CHECK: [[value3Converted:[^, ].*]][i] = int32([[value3Int64]][i])
  // CHECK: [[value4:[^, ].*]] := [[value3Converted]]
  // CHECK: return [[value4]]
  func.func @test_basic_emitter() -> tensor<4xi32> {
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

    %value1 = arith.constant dense<[1, 2, 3, 10]> : tensor<4xi32>
    %value2 = arith.constant dense<[10, 2, 3, 1]> : tensor<4xi32>

    %pt_raw1 = lattigo.bgv.new_plaintext %param : (!params) -> !pt
    %pt_raw2 = lattigo.bgv.new_plaintext %param : (!params) -> !pt

    %pt1 = lattigo.bgv.encode %encoder, %value1, %pt_raw1 : (!encoder, tensor<4xi32>, !pt) -> !pt
    %pt2 = lattigo.bgv.encode %encoder, %value2, %pt_raw2 : (!encoder, tensor<4xi32>, !pt) -> !pt
    %ct1 = lattigo.rlwe.encrypt %encryptor, %pt1 : (!encryptor, !pt) -> !ct
    %ct2 = lattigo.rlwe.encrypt %encryptor, %pt2 : (!encryptor, !pt) -> !ct

    %mul = func.call @compute(%evaluator, %ct1, %ct2) : (!evaluator, !ct, !ct) -> !ct

    %dec = lattigo.rlwe.decrypt %decryptor, %mul : (!decryptor, !ct) -> !pt

    %result = arith.constant dense<0> : tensor<4xi32>

    %updated = lattigo.bgv.decode %encoder, %dec, %result : (!encoder, !pt, tensor<4xi32>) -> tensor<4xi32>

    return %updated : tensor<4xi32>
  }
}

// -----


// CHECK: test_constant
// CHECK: [[v1:.*]] := int32(1)
// CHECK: [[v2:.*]] := []int32{1, 2}
// CHECK: [[v3:.*]] := []int32{2, 2, 2, 2}
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

// CHECK: test_new_evaluation_key_set_no_relin_key
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

// CHECK: test_new_evaluator_no_key_set
// CHECK: bgv.NewEvaluator([[params:[^, ].*]], nil, false)
module attributes {scheme.bgv} {
  func.func @test_new_evaluator_no_key_set(%params : !params) -> (!evaluator) {
    %evaluator = lattigo.bgv.new_evaluator %params : (!params) -> !evaluator
    return %evaluator : !evaluator
  }
}

// -----

// CHECK: func dot_product
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

// -----

module attributes {scheme.bgv} {
  // CHECK: func test_drop_level
  // CHECK-SAME: ([[evaluator:.*]] *bgv.Evaluator, [[ct:.*]] *rlwe.Ciphertext)
  func.func @test_drop_level(%evaluator: !lattigo.bgv.evaluator, %ct: !lattigo.rlwe.ciphertext) -> (!lattigo.rlwe.ciphertext) {
    // CHECK: [[ct1:[^, ]*]] := ct.CopyNew()
    // CHECK: evaluator.DropLevel([[ct1]], 2)
    %ct1 = lattigo.rlwe.drop_level_new %evaluator, %ct {levelToDrop = 2}: (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
    return %ct1 : !lattigo.rlwe.ciphertext
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func test_negate_new
  // CHECK-SAME: ([[evaluator:.*]] *bgv.Evaluator, [[ct:.*]] *rlwe.Ciphertext) (*rlwe.Ciphertext)
  func.func @test_negate_new(%evaluator: !lattigo.bgv.evaluator, %ct: !lattigo.rlwe.ciphertext) -> (!lattigo.rlwe.ciphertext) {
    // CHECK: [[ct1:[^, ]*]] := [[ct]].CopyNew()
    // CHECK: for [[i:[^, ]*]] := 0; [[i]] < len([[ct1]].Value); [[i]]++ {
    // CHECK:   [[evaluator]].GetRLWEParameters().RingQ().AtLevel([[ct1]].LevelQ()).Neg([[ct1]].Value[[[i]]], [[ct1]].Value[[[i]]])
    // CHECK: }
    %negated = lattigo.rlwe.negate_new %evaluator, %ct : (!lattigo.bgv.evaluator, !lattigo.rlwe.ciphertext) -> !lattigo.rlwe.ciphertext
    return %negated : !lattigo.rlwe.ciphertext
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func float_constant
  func.func @float_constant(%evaluator: !lattigo.bgv.evaluator, %ct: !lattigo.rlwe.ciphertext) -> f32 {
    // CHECK: [[v:[^, ]*]] := float32(7.5)
    // CHECK: return v
    %v = arith.constant 7.5 : f32
    return %v : f32
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func tensor_insert
  func.func @tensor_insert(%evaluator: !lattigo.bgv.evaluator, %ct: !lattigo.rlwe.ciphertext) -> f32 {
    // CHECK:  [[v0:[^ ]*]] := int64(5)
    // CHECK:  [[v1:[^, ]*]] := float32(7.5)
    // CHECK:  [[v2:[^ ]*]] := []float32{0, 0, 0, 0, 0, 0, 0, 0}
    // CHECK:  [[v3:[^ ]*]] := append(make([]float32, 0, len([[v2]])), v2...)
    // CHECK:  [[v3]]{{\[}}[[v0]]] = [[v1]]
    // CHECK:  return [[v1]]
    %c5 = arith.constant 5 : index
    %v = arith.constant 7.5 : f32
    %tensor = arith.constant dense<0.0> : tensor<8xf32>
    %tensor2 = tensor.insert %v into %tensor[%c5] : tensor<8xf32>
    return %v : f32
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func extract_slice
  func.func @extract_slice(%evaluator: !lattigo.bgv.evaluator, %ct: !lattigo.rlwe.ciphertext) {
  // CHECK:  [[v0:[^ ]*]] := []int32{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}
  // CHECK:  [[v1:[^ ]*]] := [3]int32{}
  // CHECK:  for [[dest:[^ ]*]] := 0; [[dest]] < 3; [[dest]] += 1 {
  // CHECK:    [[v1]]{{\[}}[[dest]]] = [[v0]]{{\[}}1 + [[dest]] * 2]
  // CHECK:  }
  // CHECK:  [[v1_final:.*]] := [[v1]]{{\[}}:]
    %c5 = arith.constant dense<5> : tensor<20xi32>
    %v = tensor.extract_slice %c5[1] [3] [2] : tensor<20xi32> to tensor<3xi32>
    return
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func splat
  func.func @splat(%evaluator: !lattigo.bgv.evaluator, %ct: !lattigo.rlwe.ciphertext) -> tensor<20xi32> {
    // CHECK:  [[c5:[^ ]*]] := int32(5)
    // CHECK:  [[v0:[^ ]*]] := []int32{[[c5]]}
    // CHECK:  [[v1:[^ ]*]] := slices.Repeat([[v0]], 20)
    %c5 = arith.constant 5 : i32
    %v = tensor.splat %c5 : tensor<20xi32>
    return %v : tensor<20xi32>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func extsi_scalar
  func.func @extsi_scalar(%evaluator: !lattigo.bgv.evaluator) {
  // CHECK:  [[v0:[^ ]*]] := int16(5)
  // CHECK:  [[v1:[^ ]*]] := int32([[v0]])
    %c5 = arith.constant 5 : i16
    %v = arith.extsi %c5 : i16 to i32
    return
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func extsi_tensor
  func.func @extsi_tensor(%evaluator: !lattigo.bgv.evaluator, %ct: !lattigo.rlwe.ciphertext) {
  // CHECK:  [[v0:[^ ]*]] := []int16{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}
  // CHECK:  [[v1:[^ ]*]] := make([]int32, 20)
  // CHECK:  for [[i:[^,]*]], [[var:[^ ]*]] := range [[v0]] {
  // CHECK:    [[v1]]{{\[}}[[i]]] = int32([[var]])
  // CHECK:  }
    %c5 = arith.constant dense<5> : tensor<20xi16>
    %v = arith.extsi %c5 : tensor<20xi16> to tensor<20xi32>
    return
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func select
  func.func @select(%evaluator: !lattigo.bgv.evaluator) {
    // CHECK:  [[true_val:[^ ]*]] := int32(5)
    // CHECK:  [[false_val:[^ ]*]] := int32(6)
    // CHECK:  [[cond:[^ ]*]] := bool(true)
    // CHECK:  var [[result:[^ ]*]] int32
    // CHECK:  if [[cond]] {
    // CHECK:    [[result]] = [[true_val]]
    // CHECK:  } else {
    // CHECK:    [[result]] = [[false_val]]
    // CHECK:  }
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %cond = arith.constant 1 : i1
    %v = arith.select %cond, %c5, %c6 : i32
    return
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func extui_scalar
  func.func @extui_scalar(%evaluator: !lattigo.bgv.evaluator) {
    // CHECK:  [[v0:[^ ]*]] := bool(true)
    // CHECK:  var [[v1:[^ ]*]] int32
    // CHECK:  if [[v0]] {
    // CHECK:    [[v1]] = int32(1)
    // CHECK:  } else {
    // CHECK:    [[v1]] = int32(0)
    // CHECK:  }
    %ctrue = arith.constant 1 : i1
    %v = arith.extui %ctrue : i1 to i32
    return
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func extui_tensor
  func.func @extui_tensor(%evaluator: !lattigo.bgv.evaluator, %ct: !lattigo.rlwe.ciphertext) {
    // CHECK:  [[v0:[^ ]*]] := []bool{true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true}
    // CHECK:  [[v1:[^ ]*]] := make([]int32, 20)
    // CHECK:  for [[i:[^,]*]], [[var:[^ ]*]] := range [[v0]] {
    // CHECK:    if [[var]] {
    // CHECK:      [[v1]]{{\[}}[[i]]] = int32(1)
    // CHECK:    } else {
    // CHECK:      [[v1]]{{\[}}[[i]]] = int32(0)
    // CHECK:    }
    // CHECK:  }
    %ctrue = arith.constant dense<true> : tensor<20xi1>
    %v = arith.extui %ctrue : tensor<20xi1> to tensor<20xi32>
    return
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func index_cast
  func.func @index_cast(%evaluator: !lattigo.bgv.evaluator) {
    // CHECK:  [[v0:[^ ]*]] := int64(5)
    // CHECK:  [[v1:[^ ]*]] := int32([[v0]])
    %c5 = arith.constant 5 : index
    %v = arith.index_cast %c5 : index to i32
    return
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func binops
  func.func @binops() -> i1 {
    // CHECK: [[c0:[^ ]*]] := int64(0)
    // CHECK: [[c1:[^ ]*]] := int64(1)
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    // CHECK: [[v2:[^ ]*]] := [[c0]] + [[c1]]
    // CHECK: [[v3:[^ ]*]] := [[c0]] % [[c1]]
    // CHECK: [[v4:[^ ]*]] := [[c0]] >= [[c1]]
    %0 = arith.addi %c0, %c1 : i64
    %1 = arith.remsi %c0, %c1 : i64
    %2 = arith.cmpi sge, %c0, %c1 : i64
    return %2 : i1
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func concat
  func.func @concat() -> tensor<32xi32> {
    // CHECK: [[v0:[^ ]*]] := []int32{1, 1, 1, 1, 1, 1, 1, 1}
    // CHECK: [[v1:[^ ]*]] := slices.Concat([[v0]], [[v0]], [[v0]], [[v0]])
    %c0 = arith.constant dense<1> : tensor<8xi32>
    %0 = tensor.concat dim(0) %c0, %c0, %c0, %c0 : (tensor<8xi32>, tensor<8xi32>, tensor<8xi32>, tensor<8xi32>) -> tensor<32xi32>
    return %0 : tensor<32xi32>
  }
}

// -----

module attributes {scheme.bgv} {
  // CHECK: func empty
  func.func @empty() -> tensor<8xi32> {
    // CHECK: [[v0:[^ ]*]] := make([]int32, 8)
    %0 = tensor.empty() : tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}
