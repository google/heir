// RUN: heir-opt --annotate-module="backend=openfhe scheme=ckks" --mlir-to-ckks='ciphertext-degree=16' --scheme-to-openfhe='entry-function=matmul' %s | heir-translate --emit-openfhe-pke | FileCheck %s

// CHECK: std::vector<CiphertextT> matmul(
// CHECK-SAME:    CryptoContextT [[v0:[^,]*]],
// CHECK-SAME:    std::vector<CiphertextT> [[v1:[^,]*]],
// CHECK-SAME:    std::vector<CiphertextT> [[v2:[^,]*]])
// CHECK-DAG:      std::vector<double> [[v3:.*]](16, 6);
// CHECK-DAG:      std::vector<double> [[v4:.*]](16, 3);
// CHECK-DAG:      std::vector<double> [[v5:.*]](16, 4);
// CHECK-DAG:      std::vector<double> [[v6:.*]](16, 2);
// CHECK-DAG:      auto [[v6_filled:.*]] = [[v6]];
// CHECK-DAG:      [[v6_filled]].push_back([[v6]]
// CHECK-DAG:      size_t [[v7:.*]] = 1;
// CHECK-DAG:      size_t [[v8:.*]] = 0;
// CHECK-DAG:      const auto& [[v9:.*]] = [[v1]][0 + 2 * (0)];
// CHECK-DAG:      const auto& [[v10:.*]] = [[v2]][0 + 2 * (0)];
// CHECK:          const auto& [[v11:.*]] = [[v0]]->MakeCKKSPackedPlaintext([[v6_filled]]);
// CHECK-NEXT:     const auto& [[v12:.*]] = [[v0]]->EvalMult([[v9]], [[v11]]);
// CHECK-NEXT:     const auto& [[v13:.*]] = [[v0]]->EvalAdd([[v10]], [[v12]]);
// CHECK-NEXT:     [[v2]][0 + 2 * (0)] = [[v13]];
// CHECK-COUNT-3:                  [[v0]]->EvalMult
// CHECK:          return

// CHECK: matmul__generate_crypto_context
// CHECK:          SetMultiplicativeDepth(1)
// CHECK: matmul__configure_crypto_context

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>} {
  func.func @matmul(%arg0: tensor<1x2xf32> {secret.secret}, %arg1: tensor<1x2xf32> {secret.secret}) -> tensor<1x2xf32> {
    %0 = arith.constant dense<[[2.0, 3.0], [4.0, 6.0]]> : tensor<2x2xf32>
    %1 = affine.for %arg2 = 0 to 1 iter_args(%arg3 = %arg1) -> (tensor<1x2xf32>) {
      %2 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %arg3) -> (tensor<1x2xf32>) {
        %3 = affine.for %arg6 = 0 to 2 iter_args(%arg7 = %arg5) -> (tensor<1x2xf32>) {
          %extracted = tensor.extract %arg0[%arg2, %arg6] : tensor<1x2xf32>
          %extracted_0 = tensor.extract %0[%arg6, %arg4] : tensor<2x2xf32>
          %extracted_1 = tensor.extract %arg7[%arg2, %arg4] : tensor<1x2xf32>
          %4 = arith.mulf %extracted, %extracted_0 : f32
          %5 = arith.addf %extracted_1, %4 : f32
          %inserted = tensor.insert %5 into %arg7[%arg2, %arg4] : tensor<1x2xf32>
          affine.yield %inserted : tensor<1x2xf32>
        }
        affine.yield %3 : tensor<1x2xf32>
      }
      affine.yield %2 : tensor<1x2xf32>
    }
    return %1 : tensor<1x2xf32>
  }
}
