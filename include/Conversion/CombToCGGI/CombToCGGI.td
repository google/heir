#ifndef INCLUDE_CONVERSION_COMBTOCGGI_COMBTOCGGI_TD_
#define INCLUDE_CONVERSION_COMBTOCGGI_COMBTOCGGI_TD_

include "mlir/Pass/PassBase.td"

def CombToCGGI : Pass<"comb-to-cggi"> {
  let summary = "Lower `comb` to `cggi` dialect.";

  let description = [{
    This pass lowers the `comb` dialect to `cggi` dialect.
  }];

  let dependentDialects = [
    "mlir::heir::comb::CombDialect",
    "mlir::heir::cggi::CGGIDialect",
  ];
}

#endif  // INCLUDE_CONVERSION_COMBTOCGGI_COMBTOCGGI_TD_
