// A simple test for loop support.
//
// Computes the function
//
//     def f(x):
//       sum = 1.0
//       for i in range(8):
//         sum = sum * x - 1.0
//       return sum
//
// In particular,
//
//   >>> np.linspace(0, 1, 8)
//   array([0.        , 0.14285714, 0.28571429, 0.42857143, 0.57142857,
//          0.71428571, 0.85714286, 1.        ])
//   >>> f(np.linspace(0, 1, 8))
//   array([-1.        , -1.16666629, -1.39989342, -1.74687019, -2.29543899,
//          -3.19507837, -4.66914279, -7.        ])

func.func @loop(%arg0: tensor<8xf32> {secret.secret}) -> tensor<8xf32> {
  %c1_f32 = arith.constant dense<1.0> : tensor<8xf32>
  %0 = affine.for %i = 0 to 8 iter_args(%sum_iter = %c1_f32) -> tensor<8xf32> {
    %2 = arith.mulf %arg0, %sum_iter : tensor<8xf32>
    %3 = arith.subf %2, %c1_f32 : tensor<8xf32>
    affine.yield %3 : tensor<8xf32>
  }
  return %0 : tensor<8xf32>
}
