// The `_site_initialize_0` extension module. The upstream
// `_mlir_libs/__init__.py` imports this automatically and calls
// `register_dialects` on the global DialectRegistry used for every new Context.

#include "lib/Bindings/Python/HEIRCAPI.h"
#include "mlir/include/mlir/Bindings/Python/Nanobind.h"  // from @llvm-project
#include "mlir/include/mlir/Bindings/Python/NanobindAdaptors.h"  // from @llvm-project

NB_MODULE(_site_initialize_0, m) {
  m.doc() = "HEIR dialect and pass registration";

  heirRegisterAllPasses();

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    heirRegisterAllDialects(registry);
  });
}
