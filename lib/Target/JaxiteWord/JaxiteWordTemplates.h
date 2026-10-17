#ifndef LIB_TARGET_JAXITEWORD_JAXITEWORDTEMPLATES_H_
#define LIB_TARGET_JAXITEWORD_JAXITEWORDTEMPLATES_H_

#include <string_view>

namespace mlir {
namespace heir {
namespace jaxiteword {

constexpr std::string_view kModulePrelude = R"python(
import jax
import jax.numpy as jnp
import key_gen
import numpy as np
from polynomial import Polynomial
import ckks_ctx as ckks

def _rescale_divisor(ctx, level):
  cd = ctx.parameters["composite_degree"]
  q = ctx.q_towers
  start = level - cd + 1
  result = 1
  for qi in q[start:level + 1]:
    result *= int(qi)
  return result

)python";

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_JAXITEWORD_JAXITEWORDTEMPLATES_H_
