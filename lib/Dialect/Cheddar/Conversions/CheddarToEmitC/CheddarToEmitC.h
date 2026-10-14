#ifndef LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_H_
#define LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"           // from @llvm-project

namespace mlir::heir {

// Attaches the cheddar `ConvertToEmitCPatternInterface`, so that
// `--convert-to-emitc` lowers cheddar ops to EmitC.
void registerCheddarConvertToEmitCInterface(DialectRegistry& registry);

#define GEN_PASS_DECL
#include "lib/Dialect/Cheddar/Conversions/CheddarToEmitC/CheddarToEmitC.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/Cheddar/Conversions/CheddarToEmitC/CheddarToEmitC.h.inc"

}  // namespace mlir::heir

#endif  // LIB_DIALECT_CHEDDAR_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_H_
