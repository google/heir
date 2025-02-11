#ifndef LIB_TARGET_JAXITE_JAXITETEMPLATES_H_
#define LIB_TARGET_JAXITE_JAXITETEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace jaxite {

constexpr std::string_view kModulePrelude = R"python(
import numpy as np

from typing import Dict, List

from jaxite.jaxite_bool import jaxite_bool
from jaxite.jaxite_lib import types

)python";

}  // namespace jaxite
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_JAXITE_JAXITETEMPLATES_H_
