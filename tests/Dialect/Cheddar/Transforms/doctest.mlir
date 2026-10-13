// RUN: heir-opt --cheddar-configure-crypto-context %s | FileCheck %s

// CHECK-LABEL: func.func @my_entry
// CHECK-LABEL: func.func @my_entry__setup()
// CHECK-SAME: -> tensor<!{{(cheddar\.)?}}context>
// CHECK-SAME: client.helper_func
// CHECK-SAME: client.interface_role = "client.setup"
// CHECK-SAME: heir.interface = {func_name = "my_entry", roles = ["client.setup"]}
// CHECK-DAG: %[[PARAMS:.*]] = cheddar.make_parameter <{{.*}}logN = 12, logScale = 45, mainPrimes = [36028797018652673, 35184372121601], auxPrimes = [1152921504606994433]{{.*}}> : !{{(cheddar\.)?}}parameter
// CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<!{{(cheddar\.)?}}context>
// CHECK: %[[CTX:.*]] = cheddar.create_context %[[PARAMS]], %[[EMPTY]] : (!{{(cheddar\.)?}}parameter, tensor<!{{(cheddar\.)?}}context>) -> tensor<!{{(cheddar\.)?}}context>
// CHECK: return %[[CTX]] : tensor<!{{(cheddar\.)?}}context>

// CHECK-LABEL: func.func @my_entry__keygen
// CHECK-SAME: (%[[CTX:.*]]: tensor<!{{(cheddar\.)?}}context>) -> (tensor<!{{(cheddar\.)?}}context>, tensor<!{{(cheddar\.)?}}user_interface>)
// CHECK-SAME: client.helper_func
// CHECK-SAME: client.interface_role = "client.keygen"
// CHECK-SAME: heir.interface = {func_name = "my_entry", roles = ["client.keygen"]}
// CHECK: %[[EMPTY_UI:.*]] = tensor.empty() : tensor<!{{(cheddar\.)?}}user_interface>
// CHECK: %[[UI:.*]] = cheddar.create_user_interface %[[CTX]], %[[EMPTY_UI]] : (tensor<!{{(cheddar\.)?}}context>, tensor<!{{(cheddar\.)?}}user_interface>) -> tensor<!{{(cheddar\.)?}}user_interface>
// CHECK-NOT: cheddar.prepare_rot_key
// CHECK-NOT: cheddar.prepare_bootstrap
// CHECK: return %[[CTX]], %[[UI]] : tensor<!{{(cheddar\.)?}}context>, tensor<!{{(cheddar\.)?}}user_interface>

// CHECK-LABEL: func.func @my_entry__configure()
// CHECK-SAME: -> (tensor<!{{(cheddar\.)?}}context>, tensor<!{{(cheddar\.)?}}user_interface>)
// CHECK-SAME: client.helper_func
// CHECK: %[[CTX:.*]] = call @my_entry__setup() : () -> tensor<!{{(cheddar\.)?}}context>
// CHECK: call @my_entry__keygen(%[[CTX]]) : (tensor<!{{(cheddar\.)?}}context>) -> (tensor<!{{(cheddar\.)?}}context>, tensor<!{{(cheddar\.)?}}user_interface>)
// CHECK: return %{{.*}} : tensor<!{{(cheddar\.)?}}context>, tensor<!{{(cheddar\.)?}}user_interface>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @my_entry(%ctx: !cheddar.context, %input: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %res = cheddar.neg %ctx, %input, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %res : tensor<!cheddar.ciphertext>
  }
}
