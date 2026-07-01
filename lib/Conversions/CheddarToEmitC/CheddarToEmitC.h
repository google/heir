#ifndef LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_H_
#define LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_H_

#include "mlir/include/mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"           // from @llvm-project

namespace mlir::heir {

// Attaches MemRefElementTypeInterface as an external (marker-only) model to
// emitc::OpaqueType. Needed so that the cheddar-to-emitc type converter can
// form `memref<Nx!emitc.opaque<...>>` as the converted form of
// `memref<Nx!cheddar.*>` after bufferization. Call once at tool startup.
void registerCheddarToEmitCExternalModels(DialectRegistry& registry);

#define GEN_PASS_DECL
#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Conversions/CheddarToEmitC/CheddarToEmitC.h.inc"

}  // namespace mlir::heir

#endif  // LIB_CONVERSIONS_CHEDDARTOEMITC_CHEDDARTOEMITC_H_
