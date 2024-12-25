#ifndef LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_TD_
#define LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_TD_

include "mlir/Pass/PassBase.td"

def ArithToCGGI : Pass<"arith-to-cggi"> {
  let summary = "Lower `arith` to `cggi` dialect.";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::cggi::CGGIDialect",
  ];
}

#endif  // LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGI_ARITHTOCGGI_TD_
