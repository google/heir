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
from ciphertext import Ciphertext
from polynomial import Polynomial
import ckks_ctx as ckks

)python";

constexpr std::string_view kEnsurePolyHelper = R"python(
def _ensure_poly(ctx, x, level=None):
  _cache = ctx._param_cache
  _r = _cache.r
  _c = _cache.c
  _m = _cache.num_q_at_level(level) if level is not None else None

  _data = x.polynomial if isinstance(x, Polynomial) else x
  _m_in = _data.shape[-1]
  if _m is None:
    _m = _m_in
  if _m > _m_in:
    raise ValueError(
        f"_ensure_poly: requested {_m} moduli but data only has {_m_in}"
    )

  if level is not None:
    _moduli = _cache.q_moduli_at_level(level)
  else:
    _moduli_src = getattr(x, "moduli", ctx.q_towers)
    if isinstance(_moduli_src, (int, np.integer)):
      _moduli_src = [int(_moduli_src)]
    _moduli = list(_moduli_src)[:_m]

  # Return a fresh wrapper even when x is already tiled: emitted add/sub paths
  # mutate the result object, so aliasing the source would violate SSA semantics.
  _out = Polynomial(
      {
          "batch": _data.shape[0],
          "num_elements": _data.shape[1],
          "degree": ctx.degree,
          "num_moduli": _m,
          "precision": 32,
          "degree_layout": (_r, _c),
      },
      {"moduli": _moduli},
  )
  _out.polynomial = _data.reshape(
      _data.shape[0], _data.shape[1], _r, _c, _m_in
  )[..., :_m]
  return _out

def _assign_poly(dst, src):
  for _attr in (
      "batch",
      "num_elements",
      "num_moduli",
      "degree",
      "precision",
      "degree_layout",
      "r",
      "c",
      "moduli",
      "moduli_array",
      "ntt_ctx",
      "shape_in_ntt_all_limbs",
  ):
    if hasattr(src, _attr):
      setattr(dst, _attr, getattr(src, _attr))
  dst.polynomial = src.polynomial
  if hasattr(src, "extend_polynomial"):
    dst.extend_polynomial = src.extend_polynomial

)python";

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_JAXITEWORD_JAXITEWORDTEMPLATES_H_
