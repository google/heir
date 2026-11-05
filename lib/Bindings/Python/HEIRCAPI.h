#ifndef LIB_BINDINGS_PYTHON_HEIRCAPI_H_
#define LIB_BINDINGS_PYTHON_HEIRCAPI_H_

// The C API for HEIR used by the Python bindings. The MLIR Python bindings
// only talk to the compiler through the MLIR C API, so every piece of HEIR
// that Python should see must be exposed here.

#include <stdint.h>

#include "mlir-c/IR.h"       // from @llvm-project
#include "mlir-c/Support.h"  // from @llvm-project

#ifdef __cplusplus
extern "C" {
#endif

/// Inserts all HEIR dialects, the upstream dialects HEIR uses, and all their
/// extensions and external interface models into the given registry (the same
/// set heir-opt supports).
MLIR_CAPI_EXPORTED void heirRegisterAllDialects(MlirDialectRegistry registry);

/// Registers all heir-opt passes and pass pipelines with the global pass
/// registry, except the Yosys-based ones. Idempotent.
MLIR_CAPI_EXPORTED void heirRegisterAllPasses(void);

/// Callback receiving the name and description of a translation.
typedef void (*HeirTranslationCallback)(MlirStringRef name,
                                        MlirStringRef description,
                                        void* userData);

/// Calls `callback` for each translation heir-translate supports.
MLIR_CAPI_EXPORTED void heirForEachTranslation(HeirTranslationCallback callback,
                                               void* userData);

/// Runs the heir-translate translation named `translation` (e.g.
/// "emit-openfhe-pke") on a copy of `op`. `options` are heir-translate command
/// line flags, e.g. "--openfhe-include-type=install-relative". On success, the
/// translated output is passed to `outputCallback`. On failure, a description
/// of the error (including any diagnostics) is passed to `errorCallback`.
///
/// Translation options are process-global state, so this function serializes
/// concurrent calls.
MLIR_CAPI_EXPORTED MlirLogicalResult
heirTranslate(MlirOperation op, MlirStringRef translation, intptr_t numOptions,
              const MlirStringRef* options, MlirStringCallback outputCallback,
              void* outputUserData, MlirStringCallback errorCallback,
              void* errorUserData);

#ifdef __cplusplus
}
#endif

#endif  // LIB_BINDINGS_PYTHON_HEIRCAPI_H_
