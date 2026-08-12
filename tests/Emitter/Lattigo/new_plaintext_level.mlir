// RUN: heir-translate %s --emit-lattigo --split-input-file | FileCheck %s

!pt = !lattigo.rlwe.plaintext
!encoder = !lattigo.ckks.encoder
!params = !lattigo.ckks.parameter

module attributes {scheme.ckks} {
  // CHECK: func Encode
  func.func @encode(%params: !params, %encoder: !encoder, %value: tensor<8xf32>) -> !pt {
    // CHECK: [[pt:[^, ].*]] := ckks.NewPlaintext([[params:[^,]*]], 2)
    %pt = lattigo.ckks.new_plaintext %params {level = 2 : i64} : (!params) -> !pt
    // CHECK: [[encoder:.*]].Encode(
    %res = lattigo.ckks.encode %encoder, %value, %pt {scale = 45} : (!encoder, tensor<8xf32>, !pt) -> !pt
    return %res : !pt
  }
}

// -----

!pt = !lattigo.rlwe.plaintext
!encoder = !lattigo.ckks.encoder
!params = !lattigo.ckks.parameter

module attributes {scheme.ckks} {
  // CHECK: func Encode
  func.func @encode(%params: !params, %encoder: !encoder, %value: tensor<8xf32>) -> !pt {
    // CHECK: [[pt:[^, ].*]] := ckks.NewPlaintext([[params:[^,]*]], [[params]].MaxLevel())
    %pt = lattigo.ckks.new_plaintext %params : (!params) -> !pt
    %res = lattigo.ckks.encode %encoder, %value, %pt {scale = 45} : (!encoder, tensor<8xf32>, !pt) -> !pt
    return %res : !pt
  }
}

// -----

!pt = !lattigo.rlwe.plaintext
!encoder = !lattigo.bgv.encoder
!params = !lattigo.bgv.parameter

module attributes {scheme.bgv} {
  // CHECK: func Encode
  func.func @encode(%params: !params, %encoder: !encoder, %value: tensor<8xi32>) -> !pt {
    // CHECK: [[pt:[^, ].*]] := bgv.NewPlaintext([[params:[^,]*]], 1)
    %pt = lattigo.bgv.new_plaintext %params {level = 1 : i64} : (!params) -> !pt
    %res = lattigo.bgv.encode %encoder, %value, %pt {scale = 0} : (!encoder, tensor<8xi32>, !pt) -> !pt
    return %res : !pt
  }
}
