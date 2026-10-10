#ifndef LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
#define LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_

include "mlir/Pass/PassBase.td"

def CheddarEmitCBoundary : Pass<"cheddar-emitc-boundary", "::mlir::ModuleOp"> {
  let summary = "Re-type CHEDDAR payload buffer arguments as C++ references";

  let description = [{
    Runs after `--convert-to-emitc`. A function cannot carry an `emitc.lvalue`
    argument, so each scalar payload buffer argument is re-typed to a C++
    reference (`const T&` if only read, `T&` if written). Payload array
    arguments stay C arrays (`T name[N]`), or become `const T name[N]` when only
    read. Calls to re-typed functions are re-emitted as `emitc.call_opaque`.

    The pass also emits, once per module, the standard and CHEDDAR headers,
    `using namespace cheddar;` / `using word = uint64_t;`, and the definition
    of the small `heir::getEncoder` helper that the lowering of
    `cheddar.get_encoder` calls, so the translated C++ compiles as-is.

    (* example filepath=tests/Dialect/Cheddar/Conversions/CheddarToEmitC/doctest.mlir *)
  }];

  // The prelude, helper definitions and re-emitted calls are all EmitC ops.
  let dependentDialects = ["::mlir::emitc::EmitCDialect"];
}

#endif  // LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_TD_
