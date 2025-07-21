#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace openfhe {

constexpr std::string_view kSourceRelativeOpenfheImport = R"cpp(
#include "src/pke/include/openfhe.h"  // from @openfhe
)cpp";
constexpr std::string_view kInstallationRelativeOpenfheImport = R"cpp(
#include "openfhe/pke/openfhe.h"  // from @openfhe
)cpp";
constexpr std::string_view kEmbeddedOpenfheImport = R"cpp(
#include "openfhe.h"
)cpp";

// clang-format off
constexpr std::string_view kModulePreludeTemplate = R"cpp(
using namespace lbcrypto;
using CiphertextT = ConstCiphertext<DCRTPoly>;
using MutableCiphertextT = Ciphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContext{0}RNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kWeightsPreludeTemplate = R"cpp(
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "include/cereal/archives/portable_binary.hpp" // from @cereal
#include "include/cereal/cereal.hpp" // from @cereal

struct Weights {
  std::map<std::string, std::vector<float>> floats;
  std::map<std::string, std::vector<double>> doubles;
  std::map<std::string, std::vector<int64_t>> int64_ts;
  std::map<std::string, std::vector<int32_t>> int32_ts;
  std::map<std::string, std::vector<int16_t>> int16_ts;
  std::map<std::string, std::vector<int8_t>> int8_ts;

  template <class Archive>
  void serialize(Archive &archive) {
    archive(CEREAL_NVP(floats), CEREAL_NVP(doubles), CEREAL_NVP(int64_ts),
            CEREAL_NVP(int32_ts), CEREAL_NVP(int16_ts), CEREAL_NVP(int8_ts));
  }
};

Weights GetWeightModule(const std::string& filename) {
  Weights obj;
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  cereal::PortableBinaryInputArchive archive(file);
  archive(obj);
  file.close();
  return obj;
}
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kPybindImports = R"cpp(
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kPybindCommon = R"cpp(
using namespace lbcrypto;
namespace py = pybind11;

// Minimal bindings required for generated functions to run.
// Cf. https://pybind11.readthedocs.io/en/stable/advanced/classes.html#module-local-class-bindings
// which is a temporary workaround to allow us to have multiple compilations in
// the same python program. Better would be to cache the pybind11 module across
// calls.
void bind_common(py::module &m)
{
    py::class_<PublicKeyImpl<DCRTPoly>, std::shared_ptr<PublicKeyImpl<DCRTPoly>>>(m, "PublicKey", py::module_local())
        .def(py::init<>());
    py::class_<PrivateKeyImpl<DCRTPoly>, std::shared_ptr<PrivateKeyImpl<DCRTPoly>>>(m, "PrivateKey", py::module_local())
        .def(py::init<>());
    py::class_<KeyPair<DCRTPoly>>(m, "KeyPair", py::module_local())
        .def_readwrite("publicKey", &KeyPair<DCRTPoly>::publicKey)
        .def_readwrite("secretKey", &KeyPair<DCRTPoly>::secretKey);
    py::class_<CiphertextImpl<DCRTPoly>, std::shared_ptr<CiphertextImpl<DCRTPoly>>>(m, "Ciphertext", py::module_local())
        .def(py::init<>());
    py::class_<CryptoContextImpl<DCRTPoly>, std::shared_ptr<CryptoContextImpl<DCRTPoly>>>(m, "CryptoContext", py::module_local())
        .def(py::init<>())
        .def("KeyGen", &CryptoContextImpl<DCRTPoly>::KeyGen);
}
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kPybindModuleTemplate = R"cpp(
PYBIND11_MODULE({0}, m) {{
  bind_common(m);
)cpp";
// clang-format on

// Emit a pybind11 binding that releases the GIL for the duration of the C++
// function call.  This enables multi-threaded C++ code (e.g. OpenMP parallel
// regions inside OpenFHE) to run concurrently with the Python interpreter.
// The `py::call_guard<py::gil_scoped_release>()` helper ensures the GIL is
// relinquished on entry and re-acquired on exit.
constexpr std::string_view kPybindFunctionTemplate =
    "m.def(\"{0}\", &{0}, py::call_guard<py::gil_scoped_release>());";

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_
