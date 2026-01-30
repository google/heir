//===- SCIFRBoolDialect.td - Cornami SCIFR Boolean Dialect -*- tablegen -*-===//
//     This dialect is an exit dialect for targeting Cornami's FracTLCore(R) hardware accelerator for CGGI programs.
//===----------------------------------------------------------------------===//

// Ref: Resource Estimation of CGGI and CKKS scheme workloads on
// FracTLcore Computing Fabric, ArXiV: https://arxiv.org/abs/2510.16025, 2025.

#ifndef SCIFRBOOL_DIALECT
#define SCIFRBOOL_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Cornami SCIFR Boolean dialect definition.
//===----------------------------------------------------------------------===//

def SCIFRBool_Dialect : Dialect {
    let name = "scifrbool";
    let summary = "Cornami SCIFR Boolean dialect for FHE Applications.";
    let cppNamespace = "::mlir::cornami::scifrbool";
    let useDefaultTypePrinterParser = 1;
}


#endif // SCIFRBOOL_DIALECT
