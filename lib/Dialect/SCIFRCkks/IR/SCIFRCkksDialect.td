//===- SCIFRCkksDialect.td - Cornami SCIFR CKKS Arithmetic Dialect -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef SCIFRCKKS_DIALECT
#define SCIFRCKKS_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Cornami SCIFR CKKS dialect definition.
//===----------------------------------------------------------------------===//

def SCIFRCkks_Dialect : Dialect {
    let name = "scifrckks";
    let summary = "Cornami SCIFR CKKS dialect for FHE Applications.";
    let description = [{
MLIR Dialect for Cornami SCIFR CKKS Dialect
    Cornami SCIFR CKKS dialect
    Concepts
        Section
            Contains Operators and HBM Regions and memory transfers operations
            Section can be reduced to FA (for executable) or default (which is not executable)
        Operator
            Operators are from AppStream FHE or FractlsBase only
    }];
    let cppNamespace = "::mlir::scifrckks";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// SCIFRCkks Attributes.
//===----------------------------------------------------------------------===//

class SCIFRCkks_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SCIFRCkks_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

#endif // SCIFRCKKS_DIALECT
