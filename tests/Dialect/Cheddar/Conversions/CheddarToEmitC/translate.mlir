// RUN: heir-opt --cheddar-to-emitc %s | heir-translate --mlir-to-cpp | FileCheck %s

// Pins the C++ emitted for the full --cheddar-to-emitc pipeline.

!ciphertext = !cheddar.ciphertext
!plaintext = !cheddar.plaintext
!context = !cheddar.context
!encoder = !cheddar.encoder
!evk_map = !cheddar.evk_map
!user_interface = !cheddar.user_interface

// The message is wrapped in a std::vector<Complex> in a single expression.
// CHECK: void encode(const Encoder<word>& [[ENC:v[0-9]+]], double* [[MSG:v[0-9]+]], Plaintext<word>& [[PT:v[0-9]+]])
// CHECK: std::vector<Complex> [[VEC:v[0-9]+]] = std::vector<Complex>([[MSG]], [[MSG]] + 4);
// CHECK: double [[SCALE:v[0-9]+]] = [[ENC]].GetScale(5);
// CHECK: [[ENC]].Encode([[PT]], 5, [[SCALE]], [[VEC]]);
func.func @encode(%enc: !encoder, %msg: tensor<4xf64>) -> tensor<!plaintext> {
  %d = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %enc, %msg, %d <{level = 5 : i64}> : (!encoder, tensor<4xf64>, tensor<!plaintext>) -> tensor<!plaintext>
  return %pt : tensor<!plaintext>
}

// Decode reads the real parts straight out of the decoded vector: no pointer
// to it and no per-element copy of it.
// CHECK: void decode(const Encoder<word>& [[ENC:v[0-9]+]], const Plaintext<word>& [[PT:v[0-9]+]], float* [[DST:v[0-9]+]])
// CHECK: std::vector<Complex> [[VEC:v[0-9]+]];
// CHECK: [[ENC]].Decode([[VEC]], [[PT]]);
// CHECK-NOT: &[[VEC]]
// CHECK: for (size_t [[I:[a-z]+[0-9]+]] = {{.*}}; [[I]] < {{.*}}; [[I]] += {{.*}}) {
// CHECK-NEXT: float [[RE:v[0-9]+]] = std::real([[VEC]].at([[I]]));
// CHECK-NEXT: [[DST]][[[I]]] = [[RE]];
// CHECK-NOT: std::vector<Complex> v{{[0-9]+}} = [[VEC]];
func.func @decode(%enc: !encoder, %pt: tensor<!plaintext>, %dst: tensor<4xf32>) -> tensor<4xf32> {
  %msg = cheddar.decode %enc, %pt, %dst : (!encoder, tensor<!plaintext>, tensor<4xf32>) -> tensor<4xf32>
  return %msg : tensor<4xf32>
}

// The rotation key is looked up on the EvkMap by reference.
// CHECK: void rotate(Context<word>* [[CTX:v[0-9]+]], const EvkMap<word>& [[EVK:v[0-9]+]], const Ciphertext<word>& [[CT:v[0-9]+]], Ciphertext<word>& [[OUT:v[0-9]+]])
// CHECK: const EvaluationKey<word>& [[KEY:v[0-9]+]] = [[EVK]].GetRotationKey(5);
// CHECK: [[CTX]]->HRot([[OUT]], [[CT]], [[KEY]], 5);
func.func @rotate(%ctx: !context, %evk: !evk_map, %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d = tensor.empty() : tensor<!ciphertext>
  %r = cheddar.hrot %ctx, %evk, %ct, %d <{static_distance = 5 : i64}> : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}
