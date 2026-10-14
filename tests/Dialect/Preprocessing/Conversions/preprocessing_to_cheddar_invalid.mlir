// RUN: heir-opt --preprocessing-to-cheddar --split-input-file --verify-diagnostics %s

// Move-only preprocessing loads must be borrowed directly by a Cheddar DPS
// input operand, rather than escaped or returned.
func.func @move_only_load_not_dps(%encoder: !cheddar.encoder, %input: tensor<4xf64>) -> tensor<!cheddar.plaintext> {
  %storage = preprocessing.empty : !preprocessing.storage<tensor<!cheddar.plaintext>>
  %empty0 = tensor.empty() : tensor<!cheddar.plaintext>
  %arg0 = cheddar.encode %encoder, %input, %empty0 <level = 1 : i64> : (!cheddar.encoder, tensor<4xf64>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  preprocessing.store %arg0, %storage[] site 0 <tensor<!cheddar.plaintext>> : tensor<!cheddar.plaintext>, !preprocessing.storage<tensor<!cheddar.plaintext>>
  // expected-error @below {{move-only preprocessing loads must be borrowed by a Cheddar DPS input operand}}
  // expected-error @below {{failed to legalize operation 'preprocessing.load'}}
  %pt = preprocessing.load %storage[] site 0 <tensor<!cheddar.plaintext>> : !preprocessing.storage<tensor<!cheddar.plaintext>>, tensor<!cheddar.plaintext>
  return %pt : tensor<!cheddar.plaintext>
}

// -----

// Move-only preprocessing loads cannot be passed as a DPS init (output) operand.
func.func @move_only_load_dps_init(%ctx: !cheddar.context, %encoder: !cheddar.encoder, %ct: tensor<!cheddar.ciphertext>, %input: tensor<4xf64>) {
  %storage = preprocessing.empty : !preprocessing.storage<tensor<!cheddar.plaintext>>
  %empty0 = tensor.empty() : tensor<!cheddar.plaintext>
  %arg0 = cheddar.encode %encoder, %input, %empty0 <level = 1 : i64> : (!cheddar.encoder, tensor<4xf64>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  preprocessing.store %arg0, %storage[] site 0 <tensor<!cheddar.plaintext>> : tensor<!cheddar.plaintext>, !preprocessing.storage<tensor<!cheddar.plaintext>>
  // expected-error @below {{move-only preprocessing loads must be borrowed by a Cheddar DPS input operand}}
  // expected-error @below {{failed to legalize operation 'preprocessing.load'}}
  %pt = preprocessing.load %storage[] site 0 <tensor<!cheddar.plaintext>> : !preprocessing.storage<tensor<!cheddar.plaintext>>, tensor<!cheddar.plaintext>
  %out = tensor.empty() : tensor<!cheddar.plaintext>
  %result = cheddar.encode %encoder, %input, %pt <level = 1 : i64> : (!cheddar.encoder, tensor<4xf64>, tensor<!cheddar.plaintext>) -> tensor<!cheddar.plaintext>
  return
}
