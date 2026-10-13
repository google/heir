// RUN: heir-opt --cheddar-configure-crypto-context --split-input-file --verify-diagnostics %s

// expected-error @+1 {{Module missing either cheddar.parameter_set or ckks.schemeParam attribute}}
module {
  func.func @test_missing_params(%ctx: !cheddar.context, %input: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %res = cheddar.neg %ctx, %input, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %res : tensor<!cheddar.ciphertext>
  }
}
