#ifndef LIB_UTILS_CHEDDARREF_PARAMS_H_
#define LIB_UTILS_CHEDDARREF_PARAMS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lib/Utils/CheddarRef/MontArith.h"

namespace mlir {
namespace heir {
namespace cheddar_ref {

/// Scale information parsed from parameter configuration without floating-point
/// arithmetic.
struct ScaleInfo {
  int log_default_scale = 0;
  int default_encryption_level = 0;
  int max_level = 0;
};

/// Plain struct representing loaded Cheddar parameters for word width W (32 or
/// 64).
template <unsigned W>
struct Params {
  static_assert(W == 32 || W == 64, "Params only supports W=32 or W=64");

  using Traits = MontArithTraits<W>;
  using Word = typename Traits::Word;
  using SignedWord = typename Traits::SignedWord;

  int word_size = W;
  int log_n = 0;
  size_t degree = 0;
  int alpha = 0;
  int dnum = 0;
  int L = 0;

  std::vector<Word> main_primes;
  std::vector<Word> aux_primes;
  std::vector<Word> ter_primes;

  /// All primes in Cheddar's standard limb order:
  /// ter_primes[num_ter - 1 ... 0], main_primes[0 ... num_main - 1],
  /// aux_primes[0 ... num_aux - 1].
  std::vector<Word> all_primes;

  std::vector<std::pair<int, int>> level_config;
  std::pair<int, int> additional_base = {0, 0};

  int dense_hamming_weight = 0;
  int sparse_hamming_weight = 0;

  ScaleInfo scale_info;

  /// Load and parse parameters from JSON file at `path`.
  static Params<W> LoadFromFile(const std::string& path);

  /// Parse parameters from JSON string.
  static Params<W> Parse(std::string_view json_str);
};

using Params32 = Params<32>;
using Params64 = Params<64>;

}  // namespace cheddar_ref
}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_CHEDDARREF_PARAMS_H_
