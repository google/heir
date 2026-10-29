#include "lib/Utils/CheddarRef/Params.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Error.h"  // from @llvm-project
#include "llvm/include/llvm/Support/JSON.h"   // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar_ref {

template <unsigned W>
Params<W> Params<W>::LoadFromFile(const std::string& path) {
  std::ifstream file(path);
  assert(file.is_open() && "Failed to open parameter JSON file");
  std::stringstream buffer;
  buffer << file.rdbuf();
  return Parse(buffer.str());
}

template <unsigned W>
Params<W> Params<W>::Parse(std::string_view json_str) {
  auto parsed = llvm::json::parse(json_str);
  if (!parsed) {
    llvm::report_fatal_error(parsed.takeError());
  }
  const auto* obj = parsed->getAsObject();
  assert(obj && "Expected JSON root to be an object");

  Params<W> p;
  p.word_size = W;

  auto log_degree = obj->getInteger("log_degree");
  assert(log_degree.has_value() && "Missing log_degree");
  p.log_n = static_cast<int>(*log_degree);
  p.degree = size_t(1) << p.log_n;

  auto parse_words = [&](llvm::StringRef key) -> std::vector<Word> {
    std::vector<Word> words;
    const auto* arr = obj->getArray(key);
    if (!arr) return words;
    words.reserve(arr->size());
    for (const auto& elem : *arr) {
      auto val = elem.getAsUINT64();
      assert(val.has_value() &&
             "Expected prime to be an unsigned 64-bit integer");
      words.push_back(static_cast<Word>(*val));
    }
    return words;
  };

  p.main_primes = parse_words("main_primes");
  assert(!p.main_primes.empty() && "Missing or empty main_primes");

  p.aux_primes = parse_words("auxiliary_primes");
  assert(!p.aux_primes.empty() && "Missing or empty auxiliary_primes");

  p.ter_primes = parse_words("terminal_primes");

  p.alpha = static_cast<int>(p.aux_primes.size());
  p.L = static_cast<int>(p.main_primes.size() + p.ter_primes.size());
  p.dnum = (p.alpha > 0) ? (p.L + p.alpha - 1) / p.alpha : 0;

  // Build all_primes in Cheddar limb order:
  // ter_primes[num_ter - 1 ... 0], main_primes[0 ... num_main - 1],
  // aux_primes[0 ... num_aux - 1]
  p.all_primes.reserve(p.ter_primes.size() + p.main_primes.size() +
                       p.aux_primes.size());
  for (int i = static_cast<int>(p.ter_primes.size()) - 1; i >= 0; --i) {
    p.all_primes.push_back(p.ter_primes[i]);
  }
  for (Word prime : p.main_primes) {
    p.all_primes.push_back(prime);
  }
  for (Word prime : p.aux_primes) {
    p.all_primes.push_back(prime);
  }

  if (const auto* arr = obj->getArray("level_config")) {
    for (const auto& elem : *arr) {
      const auto* pair = elem.getAsArray();
      assert(pair && pair->size() == 2 &&
             "Expected level_config entry to be a pair");
      p.level_config.emplace_back(static_cast<int>(*(*pair)[0].getAsInteger()),
                                  static_cast<int>(*(*pair)[1].getAsInteger()));
    }
  }

  if (const auto* arr = obj->getArray("additional_base")) {
    assert(arr->size() == 2 && "Expected additional_base to be a pair");
    p.additional_base = {static_cast<int>(*(*arr)[0].getAsInteger()),
                         static_cast<int>(*(*arr)[1].getAsInteger())};
  }

  if (auto hw = obj->getInteger("dense_hamming_weight")) {
    p.dense_hamming_weight = static_cast<int>(*hw);
  } else {
    p.dense_hamming_weight = (p.log_n > 0) ? (1 << (p.log_n - 1)) : 0;
  }

  if (auto hw = obj->getInteger("sparse_hamming_weight")) {
    p.sparse_hamming_weight = static_cast<int>(*hw);
  } else {
    p.sparse_hamming_weight = 32;
  }

  if (auto scale = obj->getInteger("log_default_scale")) {
    p.scale_info.log_default_scale = static_cast<int>(*scale);
  }
  if (auto enc_lvl = obj->getInteger("default_encryption_level")) {
    p.scale_info.default_encryption_level = static_cast<int>(*enc_lvl);
  }
  p.scale_info.max_level =
      p.level_config.empty() ? 0 : static_cast<int>(p.level_config.size()) - 1;

  return p;
}

template struct Params<32>;
template struct Params<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir
