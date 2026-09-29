#ifndef LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
#define LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_

include "mlir/Pass/PassBase.td"

def CheddarEmitCBoundary : Pass<"cheddar-emitc-boundary", "::mlir::ModuleOp"> {
  let summary = "Re-type CHEDDAR payload buffer arguments as C++ references";

  let description = [{
    Runs after `--convert-to-emitc`. A function cannot carry an `emitc.lvalue`
    argument, so each payload buffer argument is re-typed to a C++ reference
    (`const T&` if only read, `T&` if written), calls to such functions are
    re-emitted as `emitc.call_opaque`, the standard and CHEDDAR headers plus
    `using namespace cheddar;` / `using word = uint64_t;` are emitted so the
    translated C++ compiles as-is.

    The pass also emits the definitions of the small `heir::` helpers that the
    lowering of `cheddar.get_encoder` and `cheddar.get_mult_key` calls.

    (* example filepath=tests/Dialect/Cheddar/Conversions/CheddarToEmitC/doctest.mlir *)
  }];

  // `--convert-to-emitc=filter-dialects=...` requires the listed dialects to be
  // loaded; this pass always follows it in the pipeline, so load them here.
  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::heir::cheddar::CheddarDialect",
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
