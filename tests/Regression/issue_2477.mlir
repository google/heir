// RUN: heir-opt %s --verify-diagnostics

"builtin.module"() ({
  "func.func"() <{function_type=(!secret.secret<i16>, !secret.secret<i16>) -> !secret.secret<i16>, sym_name="f"}> ({
    ^bb0(%a0: !secret.secret<i16>, %a1: !secret.secret<i16>):
      // expected-error@+1 {{op Number of operands to generic op does not match number of block arguments in the body}}
      %0 = "secret.generic"(%a0, %a1) ({
        ^bb0(%b0: i16, %b1: i16, %b2: i16, %b3: i16):
          "secret.yield"(%b0) : (i16) -> ()
      }) : (!secret.secret<i16>, !secret.secret<i16>) -> !secret.secret<i16>
      "func.return"(%0) : (!secret.secret<i16>) -> ()
  }) : () -> ()
}) : () -> ()
