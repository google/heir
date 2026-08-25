// RUN: heir-translate %s --emit-lattigo-interface --package-name=main | FileCheck %s
// RUN: heir-translate %s --emit-lattigo-interface --package-name=main --interface-prefix=Model | FileCheck %s --check-prefix=PREFIX

// A fixed prefix lets a harness be written against the interface without
// knowing the entry function's name.
// PREFIX: type ModelContext struct {
// PREFIX: func ModelSetup() *ModelContext {
// PREFIX: func (ctx *ModelContext) Encrypt(inputs [][]float64) ModelEncrypted {

!ct = !lattigo.rlwe.ciphertext
!pt = !lattigo.rlwe.plaintext
!params = !lattigo.ckks.parameter
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!encryptor = !lattigo.rlwe.encryptor<publicKey = true>
!decryptor = !lattigo.rlwe.decryptor
!boot_evaluator = !lattigo.ckks.bootstrapping_evaluator
!lintrans = !lattigo.ckks.linear_transformation

module attributes {scheme.ckks, backend.lattigo} {
  // The bootstrapping evaluator is only in the context because __configure
  // returns it; the facade's method signatures do not change with it.
  // CHECK: type MainContext struct {
  // CHECK-NEXT: BootEvaluator *bootstrapping.Evaluator
  // CHECK-NEXT: Evaluator *ckks.Evaluator
  // CHECK-NEXT: Params ckks.Parameters
  // CHECK-NEXT: Encoder *ckks.Encoder
  // CHECK-NEXT: Encryptor *rlwe.Encryptor
  // CHECK-NEXT: Decryptor *rlwe.Decryptor
  // CHECK: func MainSetup() *MainContext {
  // CHECK: ctx.BootEvaluator, ctx.Evaluator, ctx.Params, ctx.Encoder, ctx.Encryptor, ctx.Decryptor = Main__configure()
  func.func private @main__configure() -> (!boot_evaluator, !evaluator, !params, !encoder, !encryptor, !decryptor) attributes {client.setup_func = {func_name = "main"}}

  // Two storage slices, one per distinct preprocessed element type.
  // CHECK: type MainPrepared struct {
  // CHECK-NEXT: S0 []lintrans.LinearTransformation
  // CHECK-NEXT: S1 []*rlwe.Plaintext
  // CHECK: type MainEncrypted struct {
  // CHECK-NEXT: Arg0 []*rlwe.Ciphertext
  // CHECK-NEXT: Arg1 []float32
  // CHECK-NEXT: Zero0 *rlwe.Ciphertext
  // CHECK: type MainEvaluated struct {
  // CHECK-NEXT: Res0 []*rlwe.Ciphertext

  // CHECK: func (ctx *MainContext) Encrypt(inputs [][]float64) MainEncrypted {
  // CHECK: expected 2 inputs
  // CHECK: arg0 := make([]float32, len(inputs[0]))
  // CHECK: enc.Arg0 = Main__encrypt__arg0(ctx.Evaluator, ctx.Params, ctx.Encoder, ctx.Encryptor, arg0)
  // CHECK: arg1 := make([]float32, len(inputs[1]))
  // CHECK: enc.Arg1 = arg1
  // CHECK: enc.Zero0 = Main__encrypt__zero__0(ctx.Evaluator, ctx.Params, ctx.Encoder, ctx.Encryptor)
  func.func private @main__encrypt__arg0(%evaluator: !evaluator, %params: !params, %encoder: !encoder, %encryptor: !encryptor, %arg: memref<16xf32>) -> memref<1x!ct> attributes {client.enc_func = {func_name = "main", index = 0 : i64}}

  func.func private @main__encrypt__zero__0(%evaluator: !evaluator, %params: !params, %encoder: !encoder, %encryptor: !encryptor) -> !ct attributes {client.enc_zero_func = {func_name = "main", index = 0 : i64}}

  // entry_arg_indices says entry argument 1 is the one that feeds preprocessing.
  // CHECK: func (ctx *MainContext) Preprocess(inputs [][]float64) MainPrepared {
  // CHECK: arg1 := make([]float32, len(inputs[1]))
  // CHECK: prep.S0, prep.S1 = Main__preprocessing(ctx.Params, ctx.Encoder, arg1)
  func.func private @main__preprocessing(%params: !params, %encoder: !encoder, %arg: memref<16xf32>) -> (memref<1x!lintrans>, memref<2x!pt>) attributes {server.preprocessing_func = {entry_arg_indices = array<i64: 1>, func_name = "main"}}

  // CHECK: func (ctx *MainContext) Evaluate(prep MainPrepared, enc MainEncrypted) MainEvaluated {
  // CHECK: out.Res0 = Main__preprocessed(ctx.BootEvaluator, ctx.Evaluator, ctx.Params, ctx.Encoder, enc.Arg0, enc.Arg1, enc.Zero0, prep.S0, prep.S1)
  func.func private @main__preprocessed(%boot: !boot_evaluator, %evaluator: !evaluator, %params: !params, %encoder: !encoder, %ct: memref<1x!ct>, %arg1: memref<16xf32>, %zero: !ct, %s0: memref<1x!lintrans>, %s1: memref<2x!pt>) -> memref<1x!ct> attributes {client.preprocessed_func = {func_name = "main"}, server.evaluate_func = {func_name = "main"}}

  // CHECK: func (ctx *MainContext) Decrypt(out MainEvaluated) [][]float64 {
  // CHECK: res0 := Main__decrypt__result0(ctx.Evaluator, ctx.Params, ctx.Encoder, ctx.Decryptor, out.Res0)
  // CHECK: results[0] = make([]float64, len(res0))
  func.func private @main__decrypt__result0(%evaluator: !evaluator, %params: !params, %encoder: !encoder, %decryptor: !decryptor, %ct: memref<1x!ct>) -> memref<16xf32> attributes {client.dec_func = {func_name = "main", index = 0 : i64}}

  func.func private @main(%evaluator: !evaluator, %params: !params, %encoder: !encoder, %ct: memref<1x!ct>, %arg1: memref<16xf32>, %zero: !ct {client.enc_zero_arg = {func_name = "main", index = 0 : i64}}) -> memref<1x!ct> attributes {heir.entry_func = {func_name = "main"}, heir.entry_input_types = [tensor<16xf32>, tensor<16xf32>], heir.entry_result_types = [tensor<16xf32>]}
}
