// Compiled against the CHEDDAR library (see BUILD): the op surface real
// kernels use, with ctx / user_interface / keys / evk_map as function
// arguments.

!ciphertext = !cheddar.ciphertext
!plaintext = !cheddar.plaintext
!constant = !cheddar.constant
!context = !cheddar.context
!encoder = !cheddar.encoder
!eval_key = !cheddar.eval_key
!evk_map = !cheddar.evk_map
!user_interface = !cheddar.user_interface

// Add / Sub / Mult chained on ciphertexts.
func.func @arith(%ctx: !context, %a: tensor<!ciphertext>,
                 %b: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.add %ctx, %a, %b, %d0
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %1 = cheddar.sub %ctx, %0, %b, %d1
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = tensor.empty() : tensor<!ciphertext>
  %2 = cheddar.mult %ctx, %1, %a, %d2
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %2 : tensor<!ciphertext>
}

// ct+pt and ct+const overloaded dispatch.
func.func @ct_pt_const(%ctx: !context, %ct: tensor<!ciphertext>,
                       %pt: tensor<!plaintext>, %c: tensor<!constant>)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.add_plain %ctx, %ct, %pt, %d0
      : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %1 = cheddar.sub_plain %ctx, %0, %pt, %d1
      : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = tensor.empty() : tensor<!ciphertext>
  %2 = cheddar.mult_plain %ctx, %1, %pt, %d2
      : (!context, tensor<!ciphertext>, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d3 = tensor.empty() : tensor<!ciphertext>
  %3 = cheddar.add_const %ctx, %2, %c, %d3
      : (!context, tensor<!ciphertext>, tensor<!constant>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d4 = tensor.empty() : tensor<!ciphertext>
  %4 = cheddar.mult_const %ctx, %3, %c, %d4
      : (!context, tensor<!ciphertext>, tensor<!constant>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %4 : tensor<!ciphertext>
}

// Unary ops.
func.func @unary(%ctx: !context, %ct: tensor<!ciphertext>)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.neg %ctx, %ct, %d0
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %1 = cheddar.rescale %ctx, %0, %d1
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = tensor.empty() : tensor<!ciphertext>
  %2 = cheddar.level_down %ctx, %1, %d2 {targetLevel = 2 : i64}
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %2 : tensor<!ciphertext>
}

// Relinearize / RelinearizeRescale with an evaluation-key argument.
func.func @relin(%ctx: !context, %ct: tensor<!ciphertext>,
                 %k: !cheddar.eval_key) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.relinearize %ctx, %ct, %k, %d0
      : (!context, tensor<!ciphertext>, !cheddar.eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %1 = cheddar.relinearize_rescale %ctx, %0, %k, %d1
      : (!context, tensor<!ciphertext>, !cheddar.eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %1 : tensor<!ciphertext>
}

// HMult with an evaluation-key argument.
func.func @hmult(%ctx: !context, %a: tensor<!ciphertext>,
                 %b: tensor<!ciphertext>, %k: !cheddar.eval_key)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.hmult %ctx, %a, %b, %k, %d0 {rescale = true}
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, !cheddar.eval_key, tensor<!ciphertext>)
      -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// Rotation / conjugation look the key up on the EvkMap argument.
func.func @rotations(%ctx: !context, %evk: !cheddar.evk_map,
                     %a: tensor<!ciphertext>, %b: tensor<!ciphertext>)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.hrot %ctx, %evk, %a, %d0 {static_distance = 5 : i64}
      : (!context, !cheddar.evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!ciphertext>
  %1 = cheddar.hrot_add %ctx, %evk, %0, %b, %d1 {distance = 7 : i64}
      : (!context, !cheddar.evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d2 = tensor.empty() : tensor<!ciphertext>
  %2 = cheddar.hconj %ctx, %evk, %1, %d2
      : (!context, !cheddar.evk_map, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d3 = tensor.empty() : tensor<!ciphertext>
  %3 = cheddar.hconj_add %ctx, %evk, %2, %b, %d3
      : (!context, !cheddar.evk_map, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %3 : tensor<!ciphertext>
}

// mad_unsafe with a local accumulator.
func.func @mad_local(%ctx: !context, %a: tensor<!ciphertext>,
                     %b: tensor<!ciphertext>, %c: tensor<!constant>)
    -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %acc = cheddar.add %ctx, %a, %b, %d0
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %r = cheddar.mad_unsafe %ctx, %acc, %a, %c
      : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!constant>)
      -> tensor<!ciphertext>
  return %r : tensor<!ciphertext>
}

// Bootstrapping-family ops taking an EvkMap argument (const EvkMap& at the C++
// boundary).
func.func @boot(%ctx: !cheddar.boot_context, %ct: tensor<!ciphertext>,
                %evk: !cheddar.evk_map) -> tensor<!ciphertext> {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.boot %ctx, %ct, %evk, %d0
      : (!cheddar.boot_context, tensor<!ciphertext>, !cheddar.evk_map, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}

// Encrypt / Decrypt out-param calls on the UserInterface.
func.func @encrypt_decrypt(%ui: !cheddar.user_interface, %pt: tensor<!plaintext>,
                           %ct: tensor<!ciphertext>)
    -> (tensor<!ciphertext>, tensor<!plaintext>) {
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.encrypt %ui, %pt, %d0
      : (!cheddar.user_interface, tensor<!plaintext>, tensor<!ciphertext>) -> tensor<!ciphertext>
  %d1 = tensor.empty() : tensor<!plaintext>
  %1 = cheddar.decrypt %ui, %ct, %d1
      : (!cheddar.user_interface, tensor<!ciphertext>, tensor<!plaintext>) -> tensor<!plaintext>
  return %0, %1 : tensor<!ciphertext>, tensor<!plaintext>
}

// Loop kernel: each producer writes element `i` of the out-param directly.
func.func @loop_store(%ctx: !context, %in: tensor<!ciphertext>)
    -> tensor<8x!ciphertext> {
  %out = tensor.empty() : tensor<8x!ciphertext>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%acc = %out)
      -> (tensor<8x!ciphertext>) {
    %d = tensor.empty() : tensor<!ciphertext>
    %v = cheddar.add %ctx, %in, %in, %d
        : (!context, tensor<!ciphertext>, tensor<!ciphertext>, tensor<!ciphertext>)
        -> tensor<!ciphertext>
    %ins = tensor.insert_slice %v into %acc[%i] [1] [1]
        : tensor<!ciphertext> into tensor<8x!ciphertext>
    scf.yield %ins : tensor<8x!ciphertext>
  }
  return %r : tensor<8x!ciphertext>
}

// Support values derived from the context and key material.
func.func @support_values(%ctx: !context, %ui: !user_interface,
                          %ct: tensor<!ciphertext>) -> tensor<!ciphertext> {
  %map = cheddar.get_evk_map %ui : (!user_interface) -> !evk_map
  %key = cheddar.get_mult_key %map, %ctx : (!evk_map, !context) -> !eval_key
  %d0 = tensor.empty() : tensor<!ciphertext>
  %0 = cheddar.relinearize %ctx, %ct, %key, %d0
      : (!context, tensor<!ciphertext>, !eval_key, tensor<!ciphertext>) -> tensor<!ciphertext>
  return %0 : tensor<!ciphertext>
}
