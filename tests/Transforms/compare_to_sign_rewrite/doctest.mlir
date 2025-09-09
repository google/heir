// RUN: heir-opt --compare-to-sign-rewrite %s

func.func @cmpi_sgt(%arg0: i32, %arg1: i32) -> i1 {
  %0 = arith.cmpi sgt, %arg0, %arg1 : i32
  return %0 : i1
}
