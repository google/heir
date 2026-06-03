#ifndef LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
#define LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_

include "mlir/Pass/PassBase.td"

def CheddarToEmitC : Pass<"cheddar-to-emitc"> {
  let summary = "Lower the cheddar dialect to EmitC.";

  let description = [{
    Translates each cheddar op into an out-parameter-style `Context`/
    `UserInterface` method call expressed in the EmitC dialect.
    `mlir-translate --mlir-to-cpp` renders the resulting IR as host-side
    C++ against the CHEDDAR library API.

    Bufferized loop kernels are handled in the same conversion: the SCF and
    Arith EmitC patterns are run alongside the cheddar patterns so that the
    shared type converter sees cheddar types (e.g. an `scf.for` carrying an
    `!cheddar.*` iter_arg lowers to a move-assigned `emitc.variable`, and
    loop-index values feed `memref` subscripts without a stranded
    `index -> emitc.size_t` cast).
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif  // LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
