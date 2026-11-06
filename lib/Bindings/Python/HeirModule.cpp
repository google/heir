// The `_heir` native extension module: dialect/pass registration, in-process
// access to heir-translate, and stopping pass managers between passes.

#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT(build/c++11)
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

// Runs Python's signal handlers before each pass of a pass manager, and stops
// it if one raises (e.g., KeyboardInterrupt on Ctrl-C). Otherwise, Python would
// only run them once PassManager.run returns, after the whole pipeline.
class SignalChecks {
 public:
  // A HeirPassCancellationCallback.
  static bool beforePass(void* userData) {
    return static_cast<SignalChecks*>(userData)->checkSignals();
  }

  // Returns the exception a signal handler raised (or None), and forgets it.
  nb::object takeException() {
    nb::object result = std::move(exception);
    return result.is_valid() ? result : nb::none();
  }

 private:
  bool checkSignals() {
    if (stopped.load(std::memory_order_relaxed)) return true;
    // Only check on the thread running the pass manager (nested passes may run
    // on others). Python only runs signal handlers if that's its main thread,
    // and PyErr_CheckSignals does nothing otherwise.
    if (std::this_thread::get_id() != threadId) return false;
    // PassManager.run already holds the GIL; this is only for robustness.
    nb::gil_scoped_acquire gil;
    if (PyErr_CheckSignals() == 0) return false;
    exception = nb::steal(PyErr_GetRaisedException());
    stopped.store(true, std::memory_order_relaxed);
    return true;
  }

  // The thread that will run the pass manager.
  const std::thread::id threadId = std::this_thread::get_id();
  std::atomic<bool> stopped{false};
  nb::object exception;
};

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

  nb::class_<SignalChecks>(m, "SignalChecks")
      .def("take_exception", &SignalChecks::takeException,
           "Returns the exception a signal handler raised while the pass "
           "manager ran (which stopped it), or None, and forgets it.");

  m.def(
      "check_signals_between_passes",
      [](MlirPassManager passManager) {
        auto* checks = new SignalChecks();
        heirPassManagerAddCancellationCallback(
            passManager, &SignalChecks::beforePass, checks);
        return checks;
      },
      nb::arg("pass_manager"), nb::rv_policy::take_ownership,
      // The pass manager uses the result.
      nb::keep_alive<1, 0>(),
      "Makes `pass_manager` run Python's signal handlers before each pass, if "
      "it runs on the calling thread, and stop if one raises (e.g., "
      "KeyboardInterrupt on Ctrl-C).");

  m.def(
      "translations",
      []() {
        std::vector<std::pair<std::string, std::string>> result;
        // Waits for concurrent translations (see heirTranslate).
        nb::gil_scoped_release release;
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
