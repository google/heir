// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context %s | FileCheck %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @my_entry(%ctx: !cheddar.context, %input: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %res = cheddar.neg %ctx, %input, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %res : tensor<!cheddar.ciphertext>
  }
}

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

// -----

// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context %s | FileCheck %s --check-prefix=CHECK-ROT

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [36028797018652673, 35184372121601, 35184372121603], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @test_rot(%ctx: !cheddar.context, %ui: !cheddar.user_interface, %ct: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %0 = cheddar.hrot %ctx, %ui, %ct, %out <static_distance = 5 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    %1 = cheddar.hrot_add %ctx, %ui, %0, %ct, %out <distance = 2 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    %2 = cheddar.hrot %ctx, %ui, %1, %out <static_distance = 5 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %2 : tensor<!cheddar.ciphertext>
  }
}

// CHECK-ROT-LABEL: func.func @test_rot__keygen
// CHECK-ROT: %[[EMPTY_UI:.*]] = tensor.empty() : tensor<!{{(cheddar\.)?}}user_interface>
// CHECK-ROT: %[[UI0:.*]] = cheddar.create_user_interface %{{.*}}, %[[EMPTY_UI]]
// CHECK-ROT: %[[UI1:.*]] = cheddar.prepare_rot_key %[[UI0]] <distance = 2 : i64, maxLevel = 2 : i64>
// CHECK-ROT: %[[UI2:.*]] = cheddar.prepare_rot_key %[[UI1]] <distance = 5 : i64, maxLevel = 2 : i64>
// CHECK-ROT-NOT: cheddar.prepare_rot_key
// CHECK-ROT: return %{{.*}}, %[[UI2]]

// -----

// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context=prepare-rotation-keys-at-use-levels=true %s | FileCheck %s --check-prefix=CHECK-ROT-LEVELS

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [36028797018652673, 35184372121601, 35184372121603, 35184372121605], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @test_rot_levels(%ctx: !cheddar.context, %ui: !cheddar.user_interface, %ct: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %0 = cheddar.hrot %ctx, %ui, %ct, %out <static_distance = 1 : i64> {level = 1 : i64} : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    %1 = cheddar.hrot %ctx, %ui, %0, %out <static_distance = 1 : i64> {level = 2 : i64} : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    %2 = cheddar.hrot_add %ctx, %ui, %1, %ct, %out <distance = 4 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %2 : tensor<!cheddar.ciphertext>
  }
}

// CHECK-ROT-LEVELS-LABEL: func.func @test_rot_levels__keygen
// CHECK-ROT-LEVELS: %[[UI1:.*]] = cheddar.prepare_rot_key %{{.*}} <distance = 1 : i64, maxLevel = 2 : i64>
// CHECK-ROT-LEVELS: %[[UI2:.*]] = cheddar.prepare_rot_key %[[UI1]] <distance = 4 : i64, maxLevel = 3 : i64>
// CHECK-ROT-LEVELS-NOT: cheddar.prepare_rot_key
// CHECK-ROT-LEVELS: return %{{.*}}, %[[UI2]]

// -----

// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context="dense-hamming-weight=64 sparse-hamming-weight=32" %s | FileCheck %s --check-prefix=CHECK-BOOT

module attributes {
  client.requested_slot_count = 1024 : i64,
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>
} {
  func.func @test_boot(%ctx: !cheddar.boot_context, %ct: tensor<!cheddar.ciphertext>, %evk: !cheddar.evk_map, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %0 = cheddar.boot %ctx, %ct, %evk, %out : (!cheddar.boot_context, tensor<!cheddar.ciphertext>, !cheddar.evk_map, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %0 : tensor<!cheddar.ciphertext>
  }
}

// CHECK-BOOT-LABEL: func.func @test_boot__setup()
// CHECK-BOOT-SAME: -> tensor<!{{(cheddar\.)?}}boot_context>
// CHECK-BOOT: %[[PARAMS:.*]] = cheddar.make_parameter <{{.*}}denseHammingWeight = 64{{.*}}sparseHammingWeight = 32{{.*}}> : !{{(cheddar\.)?}}parameter
// CHECK-BOOT: %[[EMPTY:.*]] = tensor.empty() : tensor<!{{(cheddar\.)?}}boot_context>
// CHECK-BOOT: %[[CTX:.*]] = cheddar.create_boot_context %[[PARAMS]], %[[EMPTY]] <numCtsLevels = 3 : i64, numStcLevels = 3 : i64> : (!{{(cheddar\.)?}}parameter, tensor<!{{(cheddar\.)?}}boot_context>) -> tensor<!{{(cheddar\.)?}}boot_context>
// CHECK-BOOT: return %[[CTX]] : tensor<!{{(cheddar\.)?}}boot_context>

// CHECK-BOOT-LABEL: func.func @test_boot__keygen
// CHECK-BOOT-SAME: (%[[CTX:.*]]: tensor<!{{(cheddar\.)?}}boot_context>) -> (tensor<!{{(cheddar\.)?}}boot_context>, tensor<!{{(cheddar\.)?}}user_interface>)
// CHECK-BOOT: %[[EMPTY_UI:.*]] = tensor.empty() : tensor<!{{(cheddar\.)?}}user_interface>
// CHECK-BOOT: %[[UI:.*]] = cheddar.create_user_interface %[[CTX]], %[[EMPTY_UI]] : (tensor<!{{(cheddar\.)?}}boot_context>, tensor<!{{(cheddar\.)?}}user_interface>) -> tensor<!{{(cheddar\.)?}}user_interface>
// CHECK-BOOT: cheddar.prepare_bootstrap %[[CTX]], %[[UI]] <numSlots = 1024 : i64> : (tensor<!{{(cheddar\.)?}}boot_context>, tensor<!{{(cheddar\.)?}}user_interface>) -> (tensor<!{{(cheddar\.)?}}boot_context>, tensor<!{{(cheddar\.)?}}user_interface>)
// CHECK-BOOT: return %{{.*}} : tensor<!{{(cheddar\.)?}}boot_context>, tensor<!{{(cheddar\.)?}}user_interface>

// CHECK-BOOT-LABEL: func.func @test_boot__configure()
// CHECK-BOOT-SAME: -> (tensor<!{{(cheddar\.)?}}boot_context>, tensor<!{{(cheddar\.)?}}user_interface>)
// CHECK-BOOT: %[[CTX:.*]] = call @test_boot__setup() : () -> tensor<!{{(cheddar\.)?}}boot_context>
// CHECK-BOOT: call @test_boot__keygen(%[[CTX]]) : (tensor<!{{(cheddar\.)?}}boot_context>) -> (tensor<!{{(cheddar\.)?}}boot_context>, tensor<!{{(cheddar\.)?}}user_interface>)
// CHECK-BOOT: return %{{.*}} : tensor<!{{(cheddar\.)?}}boot_context>, tensor<!{{(cheddar\.)?}}user_interface>

// -----

// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context %s | FileCheck %s --check-prefix=CHECK-BOOT-FULL

module attributes {
  ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>
} {
  func.func @test_boot_full(%ctx: !cheddar.boot_context, %ct: tensor<!cheddar.ciphertext>, %evk: !cheddar.evk_map, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %0 = cheddar.boot %ctx, %ct, %evk, %out : (!cheddar.boot_context, tensor<!cheddar.ciphertext>, !cheddar.evk_map, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %0 : tensor<!cheddar.ciphertext>
  }
}

// CHECK-BOOT-FULL-LABEL: func.func @test_boot_full__keygen
// CHECK-BOOT-FULL: cheddar.prepare_bootstrap %{{.*}}, %{{.*}} <numSlots = 4096 : i64>

// -----

// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context --cheddar-bufferize %s | FileCheck %s --check-prefix=CHECK-BUF

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @buf_test(%ctx: !cheddar.context, %input: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %res = cheddar.neg %ctx, %input, %out : (!cheddar.context, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %res : tensor<!cheddar.ciphertext>
  }
}

// CHECK-BUF-LABEL: func.func @buf_test__setup
// CHECK-BUF-SAME: (%[[OUT_CTX:.*]]: memref<!{{(cheddar\.)?}}context> {bufferize.result})
// CHECK-BUF: %[[PARAMS:.*]] = cheddar.make_parameter
// CHECK-BUF: cheddar.create_context %[[PARAMS]], %[[OUT_CTX]] : (!{{(cheddar\.)?}}parameter, memref<!{{(cheddar\.)?}}context>) -> ()
// CHECK-BUF: return

// CHECK-BUF-LABEL: func.func @buf_test__keygen
// CHECK-BUF-SAME: (%[[CTX:.*]]: memref<!{{(cheddar\.)?}}context>, %[[OUT_UI:.*]]: memref<!{{(cheddar\.)?}}user_interface> {bufferize.result})
// CHECK-BUF: cheddar.create_user_interface %[[CTX]], %[[OUT_UI]] : (memref<!{{(cheddar\.)?}}context>, memref<!{{(cheddar\.)?}}user_interface>) -> ()
// CHECK-BUF: return

// CHECK-BUF-LABEL: func.func @buf_test__configure
// CHECK-BUF-SAME: (%[[OUT_CTX:.*]]: memref<!{{(cheddar\.)?}}context> {bufferize.result}, %[[OUT_UI:.*]]: memref<!{{(cheddar\.)?}}user_interface> {bufferize.result})
// CHECK-BUF: %[[TMP_CTX:.*]] = memref.alloc() : memref<!{{(cheddar\.)?}}context>
// CHECK-BUF: call @buf_test__setup(%[[TMP_CTX]])
// CHECK-BUF: %[[TMP_UI:.*]] = memref.alloc() : memref<!{{(cheddar\.)?}}user_interface>
// CHECK-BUF: call @buf_test__keygen(%[[TMP_CTX]], %[[TMP_UI]])
// CHECK-BUF: memref.copy %[[TMP_CTX]], %[[OUT_CTX]]
// CHECK-BUF: memref.copy %[[TMP_UI]], %[[OUT_UI]]
// CHECK-BUF: return

// -----

// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context %s | FileCheck %s --check-prefix=CHECK-MULTI

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 12, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func private @callee(%ctx: !cheddar.context, %ui: !cheddar.user_interface, %input: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %res = cheddar.hrot %ctx, %ui, %input, %out <static_distance = 5 : i64> : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %res : tensor<!cheddar.ciphertext>
  }

  func.func @main_entry(%ctx: !cheddar.context, %ui: !cheddar.user_interface, %input: tensor<!cheddar.ciphertext>, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %res = func.call @callee(%ctx, %ui, %input, %out) : (!cheddar.context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %res : tensor<!cheddar.ciphertext>
  }
}

// CHECK-MULTI-LABEL: func.func @main_entry__keygen
// CHECK-MULTI: cheddar.prepare_rot_key %{{.*}} <distance = 5 : i64, maxLevel = 1 : i64>

// -----

// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context %s | FileCheck %s --check-prefix=CHECK-PARAM-SET

module attributes {
  cheddar.parameter_set = #cheddar.parameter_set<
    logN = 16,
    logScale = 35,
    mainPrimes = [1060765697, 1209139201],
    auxPrimes = [2147352577, 2146959361],
    terminalPrimes = [30539777],
    levelConfig = [1, 0, 2, 0],
    wordBits = 32,
    defaultEncryptionLevel = 1,
    additionalBase = [0, 0],
    denseHammingWeight = 32768,
    sparseHammingWeight = 32,
    numCtsLevels = 4,
    numStcLevels = 2
  >
} {
  func.func @test_param_set_boot(%ctx: !cheddar.boot_context, %ct: tensor<!cheddar.ciphertext>, %evk: !cheddar.evk_map, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %0 = cheddar.boot %ctx, %ct, %evk, %out : (!cheddar.boot_context, tensor<!cheddar.ciphertext>, !cheddar.evk_map, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %0 : tensor<!cheddar.ciphertext>
  }
}

// CHECK-PARAM-SET-LABEL: func.func @test_param_set_boot__setup()
// CHECK-PARAM-SET-SAME: -> tensor<!{{(cheddar\.)?}}boot_context>
// CHECK-PARAM-SET: %[[PARAMS:.*]] = cheddar.make_parameter <{{.*}}logN = 16, logScale = 35, mainPrimes = [1060765697, 1209139201], auxPrimes = [2147352577, 2146959361], terminalPrimes = [30539777], levelConfig = [1, 0, 2, 0], wordBits = 32, defaultEncryptionLevel = 1, additionalBase = [0, 0], denseHammingWeight = 32768, sparseHammingWeight = 32, boot = true, numCtsLevels = 4, numStcLevels = 2{{.*}}> : !{{(cheddar\.)?}}parameter
// CHECK-PARAM-SET: %[[EMPTY:.*]] = tensor.empty() : tensor<!{{(cheddar\.)?}}boot_context>
// CHECK-PARAM-SET: %[[CTX:.*]] = cheddar.create_boot_context %[[PARAMS]], %[[EMPTY]] <numCtsLevels = 4 : i64, numStcLevels = 2 : i64> : (!{{(cheddar\.)?}}parameter, tensor<!{{(cheddar\.)?}}boot_context>) -> tensor<!{{(cheddar\.)?}}boot_context>
// CHECK-PARAM-SET: return %[[CTX]] : tensor<!{{(cheddar\.)?}}boot_context>

// CHECK-PARAM-SET-LABEL: func.func @test_param_set_boot__keygen
// CHECK-PARAM-SET: cheddar.prepare_bootstrap %{{.*}}, %{{.*}} <numSlots = 32768 : i64>

// -----

// RUN: heir-opt --split-input-file --cheddar-configure-crypto-context="default-encryption-level=0 dense-hamming-weight=64 sparse-hamming-weight=16 num-cts-levels=5 num-stc-levels=4" %s | FileCheck %s --check-prefix=CHECK-CLI-OVERRIDE

module attributes {
  cheddar.parameter_set = #cheddar.parameter_set<
    logN = 14,
    logScale = 40,
    mainPrimes = [1000, 2000],
    auxPrimes = [3000],
    defaultEncryptionLevel = 1,
    denseHammingWeight = 128,
    sparseHammingWeight = 32,
    numCtsLevels = 3,
    numStcLevels = 3
  >
} {
  func.func @test_cli_override(%ctx: !cheddar.boot_context, %ui: !cheddar.user_interface, %ct: tensor<!cheddar.ciphertext>, %evk: !cheddar.evk_map, %out: tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext> {
    %0 = cheddar.hrot %ctx, %ui, %ct, %out <static_distance = 1 : i64> : (!cheddar.boot_context, !cheddar.user_interface, tensor<!cheddar.ciphertext>, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    %1 = cheddar.boot %ctx, %0, %evk, %out : (!cheddar.boot_context, tensor<!cheddar.ciphertext>, !cheddar.evk_map, tensor<!cheddar.ciphertext>) -> tensor<!cheddar.ciphertext>
    return %1 : tensor<!cheddar.ciphertext>
  }
}

// CHECK-CLI-OVERRIDE-LABEL: func.func @test_cli_override__setup()
// CHECK-CLI-OVERRIDE: %[[PARAMS:.*]] = cheddar.make_parameter <{{.*}}defaultEncryptionLevel = 0{{.*}}denseHammingWeight = 64{{.*}}sparseHammingWeight = 16{{.*}}numCtsLevels = 5{{.*}}numStcLevels = 4{{.*}}> : !{{(cheddar\.)?}}parameter
// CHECK-CLI-OVERRIDE: cheddar.create_boot_context %[[PARAMS]], %{{.*}} <numCtsLevels = 5 : i64, numStcLevels = 4 : i64>

// CHECK-CLI-OVERRIDE-LABEL: func.func @test_cli_override__keygen
// CHECK-CLI-OVERRIDE: cheddar.prepare_rot_key %{{.*}} <distance = 1 : i64, maxLevel = 0 : i64>
