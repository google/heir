#ifndef LIB_DIALECT_SECRET_IR_SECRETATTRIBUTES_TD_
#define LIB_DIALECT_SECRET_IR_SECRETATTRIBUTES_TD_

include "lib/Dialect/Secret/IR/SecretDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpAsmInterface.td"

class Secret_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Secret_Dialect, name, traits # [OpAsmAttrInterface]> {
  let mnemonic = attrMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    // OpAsmAttrInterface methods.
    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # attrMnemonic # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def Secret_KernelAttr : Secret_Attr<"Kernel", "kernel"> {
  let summary = "An annotation describing an implementation kernel for a given op.";
  let description = [{
    This attribute is used for two purposes:

    1. To allow the input IR to annotate fixed kernels on ops that the rest of the
    compiler must respect.
    2. To allow the layout optimization pipeline to materialize its kernel selection
    decisions to the IR.

    The `name` field corresponds to an internally-defined kernel name, and if
    `force` is set to `true`, then the kernel may not be overridden by HEIR's
    internal passes.
  }];
  let parameters = (ins "::mlir::heir::KernelName":$name, "bool":$force);
}

#endif  // LIB_DIALECT_SECRET_IR_SECRETATTRIBUTES_TD_
