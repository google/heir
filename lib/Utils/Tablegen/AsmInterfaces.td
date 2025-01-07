#ifndef LIB_UTILS_TABLEGEN_ASMUTILS_TD_
#define LIB_UTILS_TABLEGEN_ASMUTILS_TD_

include "mlir/IR/BuiltinTypeInterfaces.td"

def TypeAsmInterface: TypeInterface<"TypeAsmInterface"> {
  let cppNamespace = "::mlir::heir";

  let description = [{
    Interface for types to suggest textual asm names for SSA values with this type.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Return the suggested name for this type.",
      /*retTy=*/"::std::string",
      /*methodName=*/"suggestedName"
    >,
  ];
}

#endif  // LIB_UTILS_TABLEGEN_ASMUTILS_TD_
