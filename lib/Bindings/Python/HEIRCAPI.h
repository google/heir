#ifndef LIB_BINDINGS_PYTHON_HEIRCAPI_H_
#define LIB_BINDINGS_PYTHON_HEIRCAPI_H_

// A minimal C API for HEIR, used by the Python bindings. The MLIR Python
// bindings only talk to the compiler through the MLIR C API, so every piece of
// HEIR that Python should see must be exposed here.

#include "mlir-c/IR.h"       // from @llvm-project
#include "mlir-c/Support.h"  // from @llvm-project

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Secret, secret);

/// Inserts all HEIR dialects (and the upstream dialects HEIR uses) into the
/// given registry.
MLIR_CAPI_EXPORTED void heirRegisterAllDialects(MlirDialectRegistry registry);

/// Registers all HEIR passes and pipelines with the global pass registry, so
/// that they can be used from `PassManager.parse`.
MLIR_CAPI_EXPORTED void heirRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif  // LIB_BINDINGS_PYTHON_HEIRCAPI_H_
