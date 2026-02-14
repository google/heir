//===- SCIFRBoolDialect.td - Cornami SCIFR Boolean Dialect -*- tablegen -*-===//
//===----------------------------------------------------------------------===//

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
    let description = [{
MLIR Dialect for Cornami SCIFR Boolean Dialect | SCIFRBool
    Cornami SCIFR Boolean dialect
    Concepts
        Section
            Contains Operators and HBM Regions and memory transfers operations
            Section can be reduced to FA (for executable) or default (which is not executable)
        Operator
            Operators are from AppStream FHE or FractlsBase only
    }];
    let cppNamespace = "::mlir::scifrbool";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// SCIFRBool Attributes.
//===----------------------------------------------------------------------===//

class SCIFRBool_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SCIFRBool_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

#endif // SCIFRBOOL_DIALECT
