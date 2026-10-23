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
import finite_field
import math
from types import SimpleNamespace
import numpy as np
from polynomial import Polynomial
import ckks_ctx as ckks

def _rescale_divisor(ctx, level):
  if level <= 0:
    raise ValueError(f"cannot rescale below logical level {level}")
  source_moduli = tuple(ctx._param_cache.q_moduli_at_level(level))
  destination_moduli = tuple(ctx._param_cache.q_moduli_at_level(level - 1))
  if source_moduli[:len(destination_moduli)] != destination_moduli:
    raise ValueError("CROSS Q modulus levels are not nested prefixes")
  dropped_moduli = source_moduli[len(destination_moduli):]
  if not dropped_moduli:
    raise ValueError(f"rescale at logical level {level} drops no Q moduli")
  result = 1
  for qi in dropped_moduli:
    result *= int(qi)
  return result

_compiled_region_cache = {}
_compiled_prepare_only = False

class _CompiledPlanPrepared(Exception):
  pass

class _CompiledExecutionPlan:
  """Prepared entry execution specialized to validated ciphertext metadata."""

  def __init__(self, ctx, execute, regions, input_specs, input_levels,
               raw_input_path=None, raw_output_kind=None):
    self._context = ctx
    self._context_id = id(ctx)
    self._execute = execute
    self.regions = tuple(regions)
    self._input_levels = tuple(input_levels)
    self._parameter_cache = ctx._param_cache
    self._input_metadata = self._fingerprint_inputs(tuple(input_specs))
    self._raw_input_path = raw_input_path
    self._raw_output_kind = raw_output_kind

  def _fingerprint_inputs(self, encrypted_args):
    if len(encrypted_args) != len(self._input_levels):
      raise ValueError("compiled plan input count mismatch; prepare a new plan")
    def fingerprint(value, level):
      if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(fingerprint(item, level) for item in value))
      return _compiled_ciphertext_fingerprint(
          self._context, value, level, num_elements=2)
    return tuple(fingerprint(value, level) for value, level in
                 zip(encrypted_args, self._input_levels, strict=True))

  def _validate_inputs(self, encrypted_args):
    encrypted_args = tuple(encrypted_args)
    if (id(self._context) != self._context_id or
        self._context._param_cache is not self._parameter_cache):
      raise ValueError("compiled plan context changed; prepare a new plan")
    if self._fingerprint_inputs(encrypted_args) != self._input_metadata:
      raise ValueError("compiled input metadata changed; prepare a new plan")
    return encrypted_args

  def execute(self, encrypted_args):
    encrypted_args = self._validate_inputs(encrypted_args)
    if self._raw_input_path is None:
      return self._execute(encrypted_args)
    output, = self.wrap_raw(self._execute_raw_validated(encrypted_args))
    return [output] if self._raw_output_kind == "singleton_list" else output

  def execute_reference(self, encrypted_args):
    """Run generated orchestration for controlled correctness comparisons."""
    return self._execute(self._validate_inputs(encrypted_args))

  def execute_raw(self, encrypted_args):
    encrypted_args = self._validate_inputs(encrypted_args)
    return self._execute_raw_validated(encrypted_args)

  def _execute_raw_validated(self, encrypted_args):
    if len(self.regions) != 1:
      raise ValueError("raw dispatch requires exactly one compiled region")
    compiled, static_args, _ = self.regions[0]
    if self._raw_input_path is not None:
      ciphertext = encrypted_args
      for index in self._raw_input_path:
        ciphertext = ciphertext[index]
      return compiled(ciphertext.polynomial, static_args)
    payloads = []
    def collect(value):
      if hasattr(value, "polynomial"):
        payloads.append(value.polynomial)
      elif isinstance(value, (tuple, list)):
        for item in value:
          collect(item)
    collect(tuple(encrypted_args))
    return compiled(*payloads, static_args)

  def wrap_raw(self, raw_outputs):
    if len(self.regions) != 1:
      raise ValueError("raw wrapping requires exactly one compiled region")
    _, _, templates = self.regions[0]
    if not isinstance(raw_outputs, tuple):
      raw_outputs = (raw_outputs,)
    return tuple(
        template._clone_with_payload(payload)
        for template, payload in zip(templates, raw_outputs, strict=True))

def _synchronize_compiled_value(value):
  if hasattr(value, "block_until_ready"):
    value.block_until_ready()
  elif hasattr(value, "polynomial"):
    _synchronize_compiled_value(value.polynomial)
  elif isinstance(value, (tuple, list)):
    for item in value:
      _synchronize_compiled_value(item)

def _compiled_ciphertext_fingerprint(ctx, value, level, num_elements=None):
  expected_moduli = tuple(ctx._param_cache.q_moduli_at_level(level))
  finite_field.check_ct_operand(
      ctx._param_cache.ff_context_cls, len(expected_moduli), value,
      "compiled plan input", batch=ctx._param_cache.batch, num_elements=num_elements,
      degree_layout=ctx.degree_layout, moduli=expected_moduli)
  if tuple(value.moduli) != expected_moduli:
    raise ValueError(
        "compiled ciphertext modulus prefix does not match its HEIR level")
  rows, cols = ctx.degree_layout
  expected_shape = (
      value.batch, value.num_elements, rows, cols, len(expected_moduli))
  if tuple(value.polynomial.shape) != expected_shape:
    raise ValueError(
        f"compiled ciphertext shape {value.polynomial.shape} does not match "
        f"{expected_shape}")
  if value.polynomial.dtype != jnp.uint32:
    raise ValueError("compiled ciphertext payload must use uint32")
  scale = getattr(value, "_ckks_scale", None)
  if scale is not None and (not math.isfinite(float(scale)) or scale <= 0):
    raise ValueError("compiled ciphertext scale must be finite and positive")
  nsd = getattr(value, "_ckks_nsd", 1)
  if not isinstance(nsd, (int, np.integer)) or nsd < 1:
    raise ValueError("compiled ciphertext NSD must be a positive integer")
  return (
      level, expected_shape, str(value.polynomial.dtype),
      scale, int(nsd), tuple(value.moduli), value.degree,
      tuple(value.degree_layout), value.precision, str(value.modulus_dtype),
      type(value.ntt_ctx), type(value.ntt_ctx.ff_ctx),
      _compiled_array_device(value.polynomial))

def _compiled_plaintext_fingerprint(value):
  return (
      id(value), id(value.polynomial), tuple(value.polynomial.shape),
      str(value.polynomial.dtype), getattr(value, "_ckks_scale", None))

def _compiled_array_device(value):
  device = getattr(value, "device", None)
  return device() if callable(device) else device

def _compiled_validate_scale(scale, operation, level=None):
  if scale is None:
    return None
  location = operation if level is None else f"{operation} at logical level {level}"
  try:
    result = float(scale)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError(f"{location}: CKKS scale is not representable as a float") from error
  if not math.isfinite(result) or result <= 0:
    raise ValueError(f"{location}: CKKS scale must be finite and positive; got {result}")
  return result

def _compiled_scale_mul(lhs, rhs, operation="compiled multiplication", level=None):
  lhs = _compiled_validate_scale(lhs, operation, level)
  rhs = _compiled_validate_scale(rhs, operation, level)
  if lhs is None or rhs is None:
    return None
  return _compiled_validate_scale(lhs * rhs, operation, level)

def _compiled_validate_add(lhs_scale, lhs_nsd, rhs_scale, rhs_nsd):
  from he_ops import _validate_additive_metadata
  _validate_additive_metadata(
      SimpleNamespace(_ckks_scale=lhs_scale, _ckks_nsd=lhs_nsd),
      SimpleNamespace(_ckks_scale=rhs_scale, _ckks_nsd=rhs_nsd),
      "compiled ciphertext addition")

def _compiled_scale_rescale(scale, ctx, src_level, dst_level):
  scale = _compiled_validate_scale(scale, "compiled rescale input", src_level)
  if scale is None:
    return None
  result = scale
  for level in range(src_level, dst_level, -1):
    try:
      result /= _rescale_divisor(ctx, level)
    except OverflowError as error:
      raise ValueError(
          f"compiled rescale at logical level {level}: scale division overflow") from error
    result = _compiled_validate_scale(result, "compiled rescale output", level - 1)
  return result

def _prepare_compiled_plaintext(ctx, plaintext, level):
  _validate_compiled_plaintext(ctx, plaintext, level)
  return ctx._param_cache._prepare_plaintext_payload(
      plaintext.polynomial, level)

def _validate_compiled_plaintext(ctx, plaintext, level):
  if not isinstance(plaintext, Polynomial):
    raise TypeError("compiled plaintext must be a CROSS Polynomial")
  finite_field.check_ct_operand(
      ctx._param_cache.ff_context_cls, plaintext.num_moduli, plaintext,
      "compiled plaintext", batch=1, num_elements=1,
      degree_layout=ctx.degree_layout, moduli=plaintext.moduli)
  expected = tuple(ctx._param_cache.q_moduli_at_level(level))
  if tuple(plaintext.moduli[:len(expected)]) != expected:
    raise ValueError("compiled plaintext modulus prefix does not match its level")
  _compiled_validate_scale(
      getattr(plaintext, "_ckks_scale", None), "compiled plaintext", level)

def _prepare_compiled_add_plain(ctx, plaintext, level, ciphertext_scale):
  _validate_compiled_plaintext(ctx, plaintext, level)
  ciphertext_scale = _compiled_validate_scale(
      ciphertext_scale, "compiled add_plain ciphertext", level)
  evaluator = ctx.he_add[level]
  plaintext_scale = getattr(plaintext, "_ckks_scale", None)
  prepared = ctx._param_cache._prepare_plaintext_payload(
      plaintext.polynomial, level)
  if ciphertext_scale is None or plaintext_scale is None:
    return prepared
  ratio = evaluator._plaintext_scale_ratio_or_none(
      ciphertext_scale, plaintext_scale)
  if ratio is None:
    slots = getattr(plaintext, "_ckks_slots", None)
    if slots is None:
      raise ValueError(
          "compiled add_plain requires source slots for scale alignment")
    plaintext = ctx.encode_at_level(
        slots, level, scale=float(ciphertext_scale))
    return ctx._param_cache._prepare_plaintext_payload(
        plaintext.polynomial, level)
  if ratio != 1:
    prepared = evaluator._scale_plaintext_array(prepared, ratio)
  return prepared

def _prepare_compiled_output_template(ctx, template, level, num_elements, scale, nsd):
  """Build immutable ciphertext metadata before warm execution."""
  scale = _compiled_validate_scale(scale, "compiled output template", level)
  moduli = ctx._param_cache.q_moduli_at_level(level)
  payload = jnp.zeros(
      (template.batch, num_elements, template.r, template.c, len(moduli)),
      dtype=template.polynomial.dtype)
  result = template._clone_with_payload(
      payload,
      num_elements=num_elements,
      moduli=moduli,
      ntt_ctx=ctx._param_cache.get_sliced_ntt_q(level))
  return _wrap_compiled_ciphertext(result, payload, scale, nsd)

def _wrap_compiled_ciphertext(template, payload, scale, nsd):
  """Attach a raw result to metadata prepared with the compiled plan."""
  result = template._clone_with_payload(payload)
  if scale is None:
    result.__dict__.pop("_ckks_scale", None)
  else:
    result._ckks_scale = float(scale)
  result._ckks_nsd = int(nsd)
  return result

)python";

}  // namespace jaxiteword
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_JAXITEWORD_JAXITEWORDTEMPLATES_H_
