#ifndef LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace openfhe {

constexpr std::string_view kSourceRelativeOpenfheImport = R"cpp(
#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "src/pke/include/openfhe.h"  // from @openfhe
)cpp";
constexpr std::string_view kInstallationRelativeOpenfheImport = R"cpp(
#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "openfhe/pke/openfhe.h"  // from @openfhe
)cpp";
constexpr std::string_view kEmbeddedOpenfheImport = R"cpp(
#include <algorithm>
#include <complex>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "openfhe.h"
)cpp";

// clang-format off
constexpr std::string_view kModulePreludeTemplate = R"cpp(
using namespace lbcrypto;
using CiphertextT = Ciphertext<DCRTPoly>;
using ConstCiphertextT = ConstCiphertext<DCRTPoly>;
using CCParamsT = CCParams<CryptoContext{0}RNS>;
using CryptoContextT = CryptoContext<DCRTPoly>;
using EvalKeyT = EvalKey<DCRTPoly>;
using PlaintextT = Plaintext;
using PrivateKeyT = PrivateKey<DCRTPoly>;
using PublicKeyT = PublicKey<DCRTPoly>;
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kModuleHelperPrelude = R"cpp(
#ifndef OPENFHE_MODULE_HELPERS_
#define OPENFHE_MODULE_HELPERS_

inline std::string heir_cache_key(const char* base, CryptoContextT cc) {
  return std::string(base) + "@" +
         std::to_string(reinterpret_cast<std::uintptr_t>(cc.get()));
}

template <typename T>
inline void heir_hash_combine(std::size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
          (seed >> 2);
}

template <typename T>
inline std::string heir_cache_key(const char* base, CryptoContextT cc,
                                  const std::vector<T>& values) {
  std::size_t seed = values.size();
  for (const auto& value : values) {
    heir_hash_combine(seed, value);
  }
  return heir_cache_key(base, cc) + "#" + std::to_string(seed);
}

template <typename T>
inline std::string heir_cache_key(const char* base, CryptoContextT cc,
                                  const std::vector<T>& values,
                                  uint32_t level) {
  return heir_cache_key(base, cc, values) + "@L" + std::to_string(level);
}

struct HeirLinearTransform {
  bool initialized = false;
  uint32_t baby_step = 0;
  std::vector<uint32_t> baby_indices;
  std::vector<uint32_t> baby_rotations;
  std::vector<uint32_t> giant_steps;
  std::vector<std::vector<size_t>> giant_positions;
  std::vector<PlaintextT> plaintexts;
};

inline std::map<std::string, HeirLinearTransform>&
heir_linear_transform_cache() {
  static std::map<std::string, HeirLinearTransform> cache;
  return cache;
}

inline std::map<std::string, PlaintextT>& heir_ckks_plaintext_cache() {
  static std::map<std::string, PlaintextT> cache;
  return cache;
}

inline uint32_t heir_openfhe_level_from_orion(CryptoContextT cc,
                                              uint32_t orion_level) {
  uint32_t num_moduli =
      cc->GetCryptoParameters()->GetElementParams()->GetParams().size();
  if (orion_level >= num_moduli) {
    OPENFHE_THROW("HEIR Orion level exceeds OpenFHE modulus chain length");
  }
  return num_moduli - 1 - orion_level;
}

template <typename T>
inline std::vector<T> heir_rotate_vector(const std::vector<T>& values,
                                         int64_t rotation) {
  std::vector<T> rotated(values.size());
  int64_t slots = static_cast<int64_t>(values.size());
  if (slots == 0) {
    return rotated;
  }
  for (int64_t slot = 0; slot < slots; ++slot) {
    int64_t index = (slot + rotation) % slots;
    if (index < 0) {
      index += slots;
    }
    rotated[slot] = values[index];
  }
  return rotated;
}

inline int64_t heir_wrap_rotation(int64_t rot, int64_t slots) {
  int64_t wrapped = rot % slots;
  return wrapped < 0 ? wrapped + slots : wrapped;
}

inline uint32_t heir_find_best_bsgs_ratio(
    const std::vector<int32_t>& diagonal_indices, int64_t slots,
    int64_t log_max_ratio) {
  int64_t max_ratio = 1LL << log_max_ratio;
  for (int64_t n1 = 1; n1 < slots; n1 <<= 1) {
    std::map<int64_t, bool> rot_n1_set;
    std::map<int64_t, bool> rot_n2_set;
    for (auto rot : diagonal_indices) {
      int64_t r = heir_wrap_rotation(rot, slots);
      rot_n1_set[((r / n1) * n1) & (slots - 1)] = true;
      rot_n2_set[r & (n1 - 1)] = true;
    }
    int64_t nb_n1 = static_cast<int64_t>(rot_n1_set.size()) - 1;
    int64_t nb_n2 = static_cast<int64_t>(rot_n2_set.size()) - 1;
    if (nb_n1 > 0) {
      if (nb_n2 == max_ratio * nb_n1) return static_cast<uint32_t>(n1);
      if (nb_n2 > max_ratio * nb_n1) return static_cast<uint32_t>(n1 / 2);
    }
  }
  return 1;
}

template <typename FloatT>
inline std::vector<std::vector<std::complex<double>>>
heir_make_sparse_diagonals(const std::vector<FloatT>& flat_diagonals,
                           int64_t diagonal_count, int64_t slot_count) {
  std::vector<std::vector<std::complex<double>>> diagonals(
      diagonal_count, std::vector<std::complex<double>>(slot_count));
  for (int64_t diagonal = 0; diagonal < diagonal_count; ++diagonal) {
    for (int64_t slot = 0; slot < slot_count; ++slot) {
      diagonals[diagonal][slot] = std::complex<double>(
          flat_diagonals[diagonal * slot_count + slot], 0.0);
    }
  }
  return diagonals;
}

template <typename FloatT>
inline HeirLinearTransform heir_precompute_linear_transform(
    CryptoContextT cc, const std::vector<FloatT>& flat_diagonals,
    const std::vector<int32_t>& diagonal_indices, int64_t log_bsgs_ratio,
    uint32_t level) {
  if (diagonal_indices.empty()) {
    OPENFHE_THROW("HEIR linear transform precompute requires diagonals");
  }

  int64_t diagonal_count = static_cast<int64_t>(diagonal_indices.size());
  int64_t slot_count = static_cast<int64_t>(flat_diagonals.size()) /
                       diagonal_count;
  auto sparse_diagonals =
      heir_make_sparse_diagonals(flat_diagonals, diagonal_count, slot_count);

  HeirLinearTransform result;
  result.initialized = true;
  result.baby_step = (log_bsgs_ratio < 0)
                         ? static_cast<uint32_t>(slot_count)
                         : heir_find_best_bsgs_ratio(diagonal_indices,
                                                     slot_count,
                                                     log_bsgs_ratio);
  result.baby_indices.reserve(diagonal_count);
  result.plaintexts.reserve(diagonal_count);

  std::map<uint32_t, bool> baby_rotation_map;
  std::map<uint32_t, std::vector<size_t>> giant_to_positions;
  for (size_t position = 0; position < diagonal_indices.size(); ++position) {
    uint32_t rot = static_cast<uint32_t>(
        heir_wrap_rotation(diagonal_indices[position], slot_count));
    uint32_t giant =
        ((rot / result.baby_step) * result.baby_step) & (slot_count - 1);
    uint32_t baby = rot & (result.baby_step - 1);
    result.baby_indices.push_back(baby);
    if (baby != 0) {
      baby_rotation_map[baby] = true;
    }
    giant_to_positions[giant].push_back(position);

    auto shifted = heir_rotate_vector(sparse_diagonals[position],
                                      -static_cast<int64_t>(giant));
    result.plaintexts.push_back(cc->MakeCKKSPackedPlaintext(
        shifted, /*noiseScaleDeg=*/1, level, /*params=*/nullptr,
        /*slots=*/static_cast<uint32_t>(slot_count)));
  }

  result.baby_rotations.reserve(baby_rotation_map.size());
  for (const auto& [baby, _] : baby_rotation_map) {
    result.baby_rotations.push_back(baby);
  }
  result.giant_steps.reserve(giant_to_positions.size());
  result.giant_positions.reserve(giant_to_positions.size());
  for (auto& [giant, positions] : giant_to_positions) {
    result.giant_steps.push_back(giant);
    result.giant_positions.push_back(std::move(positions));
  }
  return result;
}

inline CiphertextT heir_eval_linear_transform(CryptoContextT cc,
                                              ConstCiphertextT ciphertext,
                                              const HeirLinearTransform& lt) {
  std::map<uint32_t, ConstCiphertextT> rotated_ciphertexts;
  rotated_ciphertexts.emplace(0, ciphertext);
  for (uint32_t baby : lt.baby_rotations) {
    rotated_ciphertexts.emplace(
        baby, cc->EvalRotate(ciphertext, static_cast<int32_t>(baby)));
  }

  CiphertextT result;
  bool result_initialized = false;
  for (size_t group = 0; group < lt.giant_steps.size(); ++group) {
    CiphertextT inner;
    bool inner_initialized = false;
    for (size_t position : lt.giant_positions[group]) {
      auto product = cc->EvalMult(rotated_ciphertexts.at(lt.baby_indices[position]),
                                  lt.plaintexts[position]);
      if (!inner_initialized) {
        inner = std::move(product);
        inner_initialized = true;
      } else {
        cc->EvalAddInPlace(inner, product);
      }
    }

    uint32_t giant = lt.giant_steps[group];
    if (giant != 0) {
      inner = cc->EvalRotate(inner, static_cast<int32_t>(giant));
    }

    if (!result_initialized) {
      result = std::move(inner);
      result_initialized = true;
    } else {
      cc->EvalAddInPlace(result, inner);
    }
  }
  return result;
}

#endif  // OPENFHE_MODULE_HELPERS_
)cpp";
// clang-format on

// clang-format off
constexpr std::string_view kWeightsPreludeTemplate = R"cpp(
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "include/cereal/archives/portable_binary.hpp"  // from @cereal
#include "include/cereal/cereal.hpp"  // from @cereal

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

CiphertextT encrypt_ckks(CryptoContextT cc, const std::vector<double>& values,
                         PublicKeyT pk) {
    auto pt = cc->MakeCKKSPackedPlaintext(values);
    return cc->Encrypt(pk, pt);
}

std::vector<double> decrypt_ckks(CryptoContextT cc, CiphertextT ct,
                                 PrivateKeyT sk, size_t length) {
    PlaintextT pt;
    cc->Decrypt(sk, ct, &pt);
    pt->SetLength(length);
    return pt->GetRealPackedValue();
}

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
  m.def("encrypt_ckks", &encrypt_ckks,
        py::call_guard<py::gil_scoped_release>());
  m.def("decrypt_ckks", &decrypt_ckks,
        py::call_guard<py::gil_scoped_release>());
)cpp";
// clang-format on

// Emit a pybind11 binding that releases the GIL for the duration of the C++
// function call.  This enables multi-threaded C++ code (e.g. OpenMP parallel
// regions inside OpenFHE) to run concurrently with the Python interpreter.
// The `py::call_guard<py::gil_scoped_release>()` helper ensures the GIL is
// relinquished on entry and re-acquired on exit.
constexpr std::string_view kPybindFunctionTemplate =
    "m.def(\"{0}\", &{0}, py::call_guard<py::gil_scoped_release>());";

// clang-format off
constexpr std::string_view KdebugHeaderImports = R"cpp(
#include <map>
#include <string>
)cpp";
// clang-format on

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHEPKETEMPLATES_H_
