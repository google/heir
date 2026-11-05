// The `_heir` native extension module: dialect/pass registration and
// in-process access to heir-translate.

#include <string>
#include <utility>
#include <vector>

#include "lib/Bindings/Python/HEIRCAPI.h"
#include "mlir/include/mlir/Bindings/Python/Nanobind.h"  // from @llvm-project
#include "mlir/include/mlir/Bindings/Python/NanobindAdaptors.h"  // from @llvm-project
#include "nanobind/stl/pair.h"    // from @nanobind
#include "nanobind/stl/string.h"  // from @nanobind
#include "nanobind/stl/vector.h"  // from @nanobind

namespace nb = nanobind;

namespace {

MlirStringRef toStringRef(const std::string& s) {
  return mlirStringRefCreate(s.data(), s.size());
}

void appendToString(MlirStringRef str, void* userData) {
  static_cast<std::string*>(userData)->append(str.data, str.length);
}

}  // namespace

NB_MODULE(_heir, m) {
  m.doc() = "Native HEIR support for the Python bindings.";

  m.def(
      "register_dialects",
      [](MlirDialectRegistry registry) { heirRegisterAllDialects(registry); },
      nb::arg("registry"),
      "Inserts all dialects supported by heir-opt into `registry`.");

  m.def(
      "register_passes", []() { heirRegisterAllPasses(); },
      "Registers all heir-opt passes and pipelines (idempotent).");

  m.def(
      "translations",
      []() {
        std::vector<std::pair<std::string, std::string>> result;
        heirForEachTranslation(
            [](MlirStringRef name, MlirStringRef description, void* userData) {
              static_cast<std::vector<std::pair<std::string, std::string>>*>(
                  userData)
                  ->emplace_back(
                      std::string(name.data, name.length),
                      std::string(description.data, description.length));
            },
            &result);
        return result;
      },
      "Returns (name, description) for each available translation.");

  m.def(
      "translate",
      [](MlirOperation op, const std::string& translation,
         const std::vector<std::string>& options) {
        std::vector<MlirStringRef> optionRefs;
        optionRefs.reserve(options.size());
        for (const std::string& option : options)
          optionRefs.push_back(toStringRef(option));
        std::string output;
        std::string error;
        MlirLogicalResult result;
        {
          nb::gil_scoped_release release;
          result =
              heirTranslate(op, toStringRef(translation), optionRefs.size(),
                            optionRefs.data(), appendToString, &output,
                            appendToString, &error);
        }
        if (mlirLogicalResultIsFailure(result)) throw std::runtime_error(error);
        return output;
      },
      nb::arg("op"), nb::arg("translation"),
      nb::arg("options") = std::vector<std::string>(),
      "Runs a heir-translate translation on (a copy of) `op` and returns the "
      "output.");
}
