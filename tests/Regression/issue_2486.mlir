// RUN: heir-opt --arith-to-mod-arith=modulus=65536 --verify-diagnostics %s

"builtin.module"() ({
  "func.func"() <{
    function_type = () -> tensor<10xf32>,
    sym_name = "test_cast"
  }> ({
    // expected-error@+1 {{arith-to-mod-arith: floating point types are not supported}}
    %0 = "arith.constant"() <{value = dense<0.000000e+00> : tensor<10xf32>}> : () -> tensor<10xf32>
    "func.return"(%0) : (tensor<10xf32>) -> ()
  }) : () -> ()
}) : () -> ()
