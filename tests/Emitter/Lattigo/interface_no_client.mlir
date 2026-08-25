// RUN: heir-translate %s --emit-lattigo-interface --package-name=main --interface-prefix=Model | FileCheck %s

// A program entered at ciphertext semantics: it never ran
// --add-client-interface, so there is no entry contract and no encryption or
// decryption helpers. The server side is still generated.

!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext
!params = !lattigo.ckks.parameter
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!encryptor = !lattigo.rlwe.encryptor<publicKey = true>
!decryptor = !lattigo.rlwe.decryptor

module attributes {scheme.ckks, backend.lattigo} {
  // CHECK: type ModelContext struct {
  // CHECK: func ModelSetup() *ModelContext {
  func.func private @model__configure() -> (!evaluator, !params, !encoder, !encryptor, !decryptor) attributes {client.setup_func = {func_name = "model"}}

  // CHECK: type ModelPrepared struct {
  // CHECK-NEXT: S0 []*rlwe.Plaintext
  // The entry's arguments come from the evaluate function, less the storage.
  // CHECK: type ModelEncrypted struct {
  // CHECK-NEXT: Arg0 *rlwe.Ciphertext
  // CHECK-NEXT: Arg1 []float64
  // CHECK-NEXT: Arg2 []float64

  // No client helpers, so the harness encrypts and decrypts for itself.
  // CHECK-NOT: func (ctx *ModelContext) Encrypt(
  // CHECK-NOT: func (ctx *ModelContext) Decrypt(
  // CHECK: func (ctx *ModelContext) EncryptedFrom(cts []*rlwe.Ciphertext, args [][]float64) ModelEncrypted {
  // CHECK: enc.Arg0 = cts[0]
  // CHECK: enc.Arg1 = arg1
  // CHECK: enc.Arg2 = arg2

  // CHECK: func (ctx *ModelContext) Preprocess(inputs [][]float64) ModelPrepared {
  // CHECK: prep.S0 = Model__preprocessing(ctx.Params, ctx.Encoder, arg2)
  func.func private @model__preprocessing(%params: !params, %encoder: !encoder, %arg: memref<16xf64>) -> memref<2x!pt> attributes {server.preprocessing_func = {entry_arg_indices = array<i64: 2>, func_name = "model"}}

  // CHECK: func (ctx *ModelContext) Evaluate(prep ModelPrepared, enc ModelEncrypted) ModelEvaluated {
  // CHECK: out.Res0 = Model__preprocessed(ctx.Evaluator, ctx.Params, ctx.Encoder, enc.Arg0, enc.Arg1, enc.Arg2, prep.S0)
  func.func private @model__preprocessed(%evaluator: !evaluator, %params: !params, %encoder: !encoder, %ct: !ct, %arg1: memref<16xf64>, %arg2: memref<16xf64>, %s0: memref<2x!pt>) -> !ct attributes {client.preprocessed_func = {func_name = "model"}, server.evaluate_func = {func_name = "model"}}
}
