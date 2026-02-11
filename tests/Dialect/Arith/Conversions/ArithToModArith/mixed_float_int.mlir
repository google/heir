// RUN: heir-opt --arith-to-mod-arith=modulus=65536 %s | FileCheck %s

module {
  // CHECK: @test_sitofp
  func.func @test_sitofp(%arg0: i32) -> f32 {
    // CHECK: %[[ENC:.*]] = mod_arith.encapsulate %arg0 : i32 -> !mod_arith.int<65536 : i64>
    // CHECK: %[[EXTRACTED:.*]] = mod_arith.extract %[[ENC]] : !mod_arith.int<65536 : i64> -> i64
    // CHECK: %[[TRUNC:.*]] = arith.trunci %[[EXTRACTED]] : i64 to i32
    // CHECK: %[[RESULT:.*]] = arith.sitofp %[[TRUNC]] : i32 to f32
    // CHECK: return %[[RESULT]] : f32
    %1 = arith.sitofp %arg0 : i32 to f32
    return %1 : f32
  }

  // CHECK: @test_fptosi
  func.func @test_fptosi(%arg0: f32) -> i32 {
    // CHECK: %[[INT:.*]] = arith.fptosi %arg0 : f32 to i32
    // CHECK: %[[EXT:.*]] = arith.extsi %[[INT]] : i32 to i64
    // CHECK: %[[ENC:.*]] = mod_arith.encapsulate %[[EXT]] : i64 -> !mod_arith.int<65536 : i64>
    // CHECK: %[[EXTRACTED:.*]] = mod_arith.extract %[[ENC]] : !mod_arith.int<65536 : i64> -> i64
    // CHECK: %[[RESULT:.*]] = arith.trunci %[[EXTRACTED]] : i64 to i32
    // CHECK: return %[[RESULT]] : i32
    %0 = arith.fptosi %arg0 : f32 to i32
    return %0 : i32
  }

  // CHECK: @test_cmpf
  func.func @test_cmpf(%arg0: f32, %arg1: f32) -> i1 {
    // CHECK: %[[CMP:.*]] = arith.cmpf oeq, %arg0, %arg1 : f32
    // CHECK: %[[EXT:.*]] = arith.extui %[[CMP]] : i1 to i64
    // CHECK: %[[ENC:.*]] = mod_arith.encapsulate %[[EXT]] : i64 -> !mod_arith.int<65536 : i64>
    // CHECK: %[[EXTRACTED:.*]] = mod_arith.extract %[[ENC]] : !mod_arith.int<65536 : i64> -> i64
    // CHECK: %[[RESULT:.*]] = arith.trunci %[[EXTRACTED]] : i64 to i1
    // CHECK: return %[[RESULT]] : i1
    %0 = arith.cmpf oeq, %arg0, %arg1 : f32
    return %0 : i1
  }

  // CHECK: @test_index_cast_to_modular
  func.func @test_index_cast_to_modular(%arg0: index) -> i32 {
    // CHECK: %[[INT:.*]] = arith.index_cast %arg0 : index to i64
    // CHECK: %[[ENC:.*]] = mod_arith.encapsulate %[[INT]] : i64 -> !mod_arith.int<65536 : i64>
    // CHECK: %[[EXTRACTED:.*]] = mod_arith.extract %[[ENC]] : !mod_arith.int<65536 : i64> -> i64
    // CHECK: %[[RESULT:.*]] = arith.trunci %[[EXTRACTED]] : i64 to i32
    // CHECK: return %[[RESULT]] : i32
    %0 = arith.index_cast %arg0 : index to i32
    return %0 : i32
  }

  // CHECK: @test_select
  func.func @test_select(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
    // CHECK: %[[CONDC:.*]] = mod_arith.encapsulate {{.*}} : i64 -> !mod_arith.int<65536 : i64>
    // CHECK: %[[EXTRACTED_COND:.*]] = mod_arith.extract %[[CONDC]] : !mod_arith.int<65536 : i64> -> i64
    // CHECK: %[[TRUNC_COND:.*]] = arith.trunci %[[EXTRACTED_COND]] : i64 to i1
    // CHECK: %[[TRUE_ENC:.*]] = mod_arith.encapsulate {{.*}} : i64 -> !mod_arith.int<65536 : i64>
    // CHECK: %[[FALSE_ENC:.*]] = mod_arith.encapsulate {{.*}} : i64 -> !mod_arith.int<65536 : i64>
    // CHECK: %[[SEL:.*]] = arith.select %[[TRUNC_COND]], %[[TRUE_ENC]], %[[FALSE_ENC]] : !mod_arith.int<65536 : i64>
    // CHECK: %[[EXTRACTED_SEL:.*]] = mod_arith.extract %[[SEL]] : !mod_arith.int<65536 : i64> -> i64
    // CHECK: %[[RESULT:.*]] = arith.trunci %[[EXTRACTED_SEL]] : i64 to i32
    // CHECK: return %[[RESULT]] : i32
    %0 = arith.select %arg0, %arg1, %arg2 : i32
    return %0 : i32
  }
}
