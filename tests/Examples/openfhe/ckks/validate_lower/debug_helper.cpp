#include "tests/Examples/openfhe/ckks/validate_lower/debug_helper.h"

#include <cassert>
#include <map>
#include <string>
#include <vector>

void __heir_debug(CryptoContextT cc, std::vector<CiphertextT> ct,
                  const std::map<std::string, std::string>& debugAttrMap) {
  auto name = debugAttrMap.at("debug.name");
  assert(name == "input_val" || name == "output_val");
}
