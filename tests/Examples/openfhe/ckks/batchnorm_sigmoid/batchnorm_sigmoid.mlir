func.func @batchnorm_sigmoid(%x: f32 {secret.secret}, %mean: f32, %var: f32, %gamma: f32, %beta: f32) -> f32 {
  %epsilon = arith.constant 1.000000e-05 : f32
  %var_eps = arith.addf %var, %epsilon : f32
  %sqrt_var_eps = math.sqrt %var_eps : f32

  %c0 = arith.constant 0.0 : f32
  %not_zero = arith.cmpf one, %sqrt_var_eps, %c0 : f32
  cf.assert %not_zero, "variance is zero"

  %inv_sqrt_var_eps = arith.divf %gamma, %sqrt_var_eps : f32
  %x_minus_mean = arith.subf %x, %mean : f32
  %normalized = arith.mulf %x_minus_mean, %inv_sqrt_var_eps : f32
  %bn_out = arith.addf %normalized, %beta : f32

  %neg_bn_out = arith.negf %bn_out : f32
  %exp = math.exp %neg_bn_out {degree = 3 : i32, domain_lower = -5.0 : f64, domain_upper = 5.0 : f64} : f32
  %c1 = arith.constant 1.0 : f32
  %denom = arith.addf %exp, %c1 : f32
  %sigmoid = arith.divf %c1, %denom {degree = 3 : i32, domain_lower = 1.0 : f64, domain_upper = 150.0 : f64} : f32

  return %sigmoid : f32
}
