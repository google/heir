// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

module attributes {backend.openfhe, scheme.ckks} {
  func.func @main() -> i32 {
    // CHECK: std::vector<float>
    %cst_9 = arith.constant dense_resource<resource1> : tensor<4xf32>
    %cst = arith.constant 1 : i32
    return %cst : i32
  }
}

{-#
  dialect_resources: {
    builtin: {
      resource1: "0x40000000CDCC8C3FCDCC0C403333534000000000"
    }
  }
#-}
