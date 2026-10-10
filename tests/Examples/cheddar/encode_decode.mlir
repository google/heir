// Compiled and linked against the CHEDDAR library (see BUILD): a small
// encode -> encrypt -> rotate/conjugate -> relinearize -> decrypt -> decode
// round trip, exercising the cleartext message bridges in real C++.

!ciphertext = !cheddar.ciphertext
!plaintext = !cheddar.plaintext
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!user_interface = !cheddar.user_interface

func.func @encode_decode(%ctx: !context, %ui: !user_interface,
                         %msg: tensor<8xf64>, %dst: tensor<8xf32>)
    -> tensor<8xf32> {
  %enc = cheddar.get_encoder %ctx : (!context) -> !encoder
  %evk = cheddar.get_evk_map %ui : (!user_interface) -> !evk_map
  %d0 = tensor.empty() : tensor<!plaintext>
  %pt = cheddar.encode %enc, %msg, %d0 <{level = 3 : i64}>
      : (!encoder, tensor<8xf64>, tensor<!plaintext>) -> tensor<!plaintext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %ct = cheddar.encrypt %ui, %pt, %d1
      : (!user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = tensor.empty() : tensor<!ciphertext>
  %rot = cheddar.hrot %ctx, %evk, %ct, %d2 <{static_distance = 1 : i64}>
      : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d3 = tensor.empty() : tensor<!ciphertext>
  %conj = cheddar.hconj %ctx, %evk, %rot, %d3
      : (!context, !evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d4 = tensor.empty() : tensor<!ciphertext>
  %prod = cheddar.mult %ctx, %conj, %ct, %d4
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %key = cheddar.get_mult_key %evk, %ctx : (!evk_map, !context) -> !eval_key
  %d5 = tensor.empty() : tensor<!ciphertext>
  %relin = cheddar.relinearize %ctx, %prod, %key, %d5
      : (!context, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d6 = tensor.empty() : tensor<!plaintext>
  %out = cheddar.decrypt %ui, %relin, %d6
      : (!user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
  %res = cheddar.decode %enc, %out, %dst
      : (!encoder, tensor<!plaintext>, tensor<8xf32>) -> tensor<8xf32>
  return %res : tensor<8xf32>
}
