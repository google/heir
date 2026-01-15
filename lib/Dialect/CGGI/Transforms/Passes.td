#ifndef LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def DecomposeOperations : Pass<"cggi-decompose-operations"> {
  let summary = "Expands CGGI operations into LWE operations and programmable bootstraps";
  let description = [{
    This pass expands high level CGGI operations (e.g. LUT2, XOR, etc.).

    If the option `expand-lincomb` is set, the expansion will continue into the
    component LWE scalar operations and a programmable bootstrap operation.
    Otherwise, the expansion will be stop at the `cggi.lut_lincomb` level. By
    default, `expand-lincomb` is true.

    For example, a LUT3 operation is composed of three LWE ciphertext inputs $c,
    b, a$ (in MSB to LSB ordering) which must be combined via the linear
    combination $4 * c + 2 * b + a$ before being fed into a programmable
    bootstrap defined by the lookup table.

    This pass supports XOR, LUT2, LUT3, and LutLincomb operations.
  }];
  let dependentDialects = ["mlir::heir::cggi::CGGIDialect"];

  let options = [
    Option<"expandLincomb", "expand-lincomb", "bool",
           /*default=*/"true", "Expand lincomb operations to the PBS and scalar level">
  ];
}

#endif  // LIB_DIALECT_CGGI_TRANSFORMS_PASSES_TD_
