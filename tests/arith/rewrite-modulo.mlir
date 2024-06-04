// RUN: heir-opt -convert-rem-to-arith-ext --split-input-file %s | FileCheck %s

func.func @test_add_rewrite(%lhs : i8, %rhs : i8) -> i8 {
  %n_lhs = arith_ext.normalised %lhs {q = 17} : i8
  %n_rhs = arith_ext.normalised %rhs {q = 17} : i8

  %cmod = arith.constant 17 : i8

  %add = arith.addi %n_lhs, %n_rhs : i8
  %res = arith.remui %add, %cmod : i8

  return %res : i8
}

// -----
