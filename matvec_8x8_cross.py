import jax
import jax.numpy as jnp
import key_gen
import numpy as np
from ciphertext import Ciphertext
from polynomial import Polynomial
import ckks_ctx as ckks


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


def matvec_identity__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> (np.ndarray, np.ndarray):
  v2 = np.full((8,), 0.000000e00, dtype=np.float32)
  v3 = np.full((8,), 1.000000e00, dtype=np.float32)
  pt = v0.encode(v2)
  pt1 = v0.encode(v3)
  v4 = [pt]
  v5 = [pt1]
  return (v4, v5)


def matvec_identity__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
) -> np.ndarray:
  v5 = 1
  v6 = 2
  v7 = 3
  v8 = 6
  v9 = 0
  pt = v3[0]
  pt1 = v4[0]
  ct = v2[0]
  ct1_arg = _ensure_poly(v0, ct, v0.max_level)
  ct1 = v0.he_rot[v0.max_level, 1].rotate(ct1_arg)
  ct2_arg = _ensure_poly(v0, ct1, v0.max_level)
  ct2_pt_ntt = (
      pt.polynomial[0, 0, :, : ct2_arg.polynomial.shape[-1]]
      .reshape(ct2_arg.r, ct2_arg.c, ct2_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct2_ptct = v0.ptct_mul[v0.max_level]
  ct2_ptct.set_plaintext(ct2_pt_ntt)
  ct2 = ct2_ptct.mul(ct2_arg, use_bat=False)
  ct3_arg = _ensure_poly(v0, ct, v0.max_level)
  ct3 = v0.he_rot[v0.max_level, 2].rotate(ct3_arg)
  ct4_arg = _ensure_poly(v0, ct3, v0.max_level)
  ct4_pt_ntt = (
      pt.polynomial[0, 0, :, : ct4_arg.polynomial.shape[-1]]
      .reshape(ct4_arg.r, ct4_arg.c, ct4_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct4_ptct = v0.ptct_mul[v0.max_level]
  ct4_ptct.set_plaintext(ct4_pt_ntt)
  ct4 = ct4_ptct.mul(ct4_arg, use_bat=False)
  ct5_arg = _ensure_poly(v0, ct, v0.max_level)
  ct5_pt_ntt = (
      pt.polynomial[0, 0, :, : ct5_arg.polynomial.shape[-1]]
      .reshape(ct5_arg.r, ct5_arg.c, ct5_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5 = ct5_ptct.mul(ct5_arg, use_bat=False)
  ct6_lhs = ct5.polynomial if hasattr(ct5, "polynomial") else ct5
  ct6_rhs = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  ct6_lhs = ct6_lhs.reshape(
      ct6_lhs.shape[0],
      ct6_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct6_lhs.shape[-1],
  )
  ct6_rhs = ct6_rhs.reshape(
      ct6_rhs.shape[0],
      ct6_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct6_rhs.shape[-1],
  )
  if ct6_lhs.shape != ct6_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct6_num_moduli = ct6_lhs.shape[-1]
  if hasattr(ct5, "moduli") and hasattr(ct2, "moduli"):
    if list(ct5.moduli)[:ct6_num_moduli] != list(ct2.moduli)[:ct6_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct6_moduli_src = getattr(ct5, "moduli", getattr(ct2, "moduli", v0.q_towers))
  if isinstance(ct6_moduli_src, (int, np.integer)):
    ct6_moduli_src = [ct6_moduli_src]
  ct6_moduli = jnp.array(
      list(ct6_moduli_src)[:ct6_num_moduli], dtype=jnp.uint64
  )
  ct6_sum = ct6_lhs.astype(jnp.uint64) + ct6_rhs.astype(jnp.uint64)
  ct6 = jnp.where(ct6_sum >= ct6_moduli, ct6_sum - ct6_moduli, ct6_sum).astype(
      jnp.uint32
  )
  ct7_lhs = ct6.polynomial if hasattr(ct6, "polynomial") else ct6
  ct7_rhs = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  ct7_lhs = ct7_lhs.reshape(
      ct7_lhs.shape[0],
      ct7_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct7_lhs.shape[-1],
  )
  ct7_rhs = ct7_rhs.reshape(
      ct7_rhs.shape[0],
      ct7_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct7_rhs.shape[-1],
  )
  if ct7_lhs.shape != ct7_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct7_num_moduli = ct7_lhs.shape[-1]
  if hasattr(ct6, "moduli") and hasattr(ct4, "moduli"):
    if list(ct6.moduli)[:ct7_num_moduli] != list(ct4.moduli)[:ct7_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct7_moduli_src = getattr(ct6, "moduli", getattr(ct4, "moduli", v0.q_towers))
  if isinstance(ct7_moduli_src, (int, np.integer)):
    ct7_moduli_src = [ct7_moduli_src]
  ct7_moduli = jnp.array(
      list(ct7_moduli_src)[:ct7_num_moduli], dtype=jnp.uint64
  )
  ct7_sum = ct7_lhs.astype(jnp.uint64) + ct7_rhs.astype(jnp.uint64)
  ct7 = jnp.where(ct7_sum >= ct7_moduli, ct7_sum - ct7_moduli, ct7_sum).astype(
      jnp.uint32
  )
  ct8_arg = _ensure_poly(v0, ct7, v0.max_level)
  ct8 = v0.he_rot[v0.max_level, 3].rotate(ct8_arg)
  ct9_arg = _ensure_poly(v0, ct6, v0.max_level)
  ct9 = v0.he_rot[v0.max_level, 6].rotate(ct9_arg)
  ct10_arg = _ensure_poly(v0, ct, v0.max_level)
  ct10_pt_ntt = (
      pt1.polynomial[0, 0, :, : ct10_arg.polynomial.shape[-1]]
      .reshape(ct10_arg.r, ct10_arg.c, ct10_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct10_ptct = v0.ptct_mul[v0.max_level]
  ct10_ptct.set_plaintext(ct10_pt_ntt)
  ct10 = ct10_ptct.mul(ct10_arg, use_bat=False)
  ct11_lhs = ct10.polynomial if hasattr(ct10, "polynomial") else ct10
  ct11_rhs = ct2.polynomial if hasattr(ct2, "polynomial") else ct2
  ct11_lhs = ct11_lhs.reshape(
      ct11_lhs.shape[0],
      ct11_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct11_lhs.shape[-1],
  )
  ct11_rhs = ct11_rhs.reshape(
      ct11_rhs.shape[0],
      ct11_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct11_rhs.shape[-1],
  )
  if ct11_lhs.shape != ct11_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct11_num_moduli = ct11_lhs.shape[-1]
  if hasattr(ct10, "moduli") and hasattr(ct2, "moduli"):
    if (
        list(ct10.moduli)[:ct11_num_moduli]
        != list(ct2.moduli)[:ct11_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct11_moduli_src = getattr(ct10, "moduli", getattr(ct2, "moduli", v0.q_towers))
  if isinstance(ct11_moduli_src, (int, np.integer)):
    ct11_moduli_src = [ct11_moduli_src]
  ct11_moduli = jnp.array(
      list(ct11_moduli_src)[:ct11_num_moduli], dtype=jnp.uint64
  )
  ct11_sum = ct11_lhs.astype(jnp.uint64) + ct11_rhs.astype(jnp.uint64)
  ct11 = jnp.where(
      ct11_sum >= ct11_moduli, ct11_sum - ct11_moduli, ct11_sum
  ).astype(jnp.uint32)
  ct12_lhs = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  ct12_rhs = ct8.polynomial if hasattr(ct8, "polynomial") else ct8
  ct12_lhs = ct12_lhs.reshape(
      ct12_lhs.shape[0],
      ct12_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct12_lhs.shape[-1],
  )
  ct12_rhs = ct12_rhs.reshape(
      ct12_rhs.shape[0],
      ct12_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct12_rhs.shape[-1],
  )
  if ct12_lhs.shape != ct12_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct12_num_moduli = ct12_lhs.shape[-1]
  if hasattr(ct4, "moduli") and hasattr(ct8, "moduli"):
    if list(ct4.moduli)[:ct12_num_moduli] != list(ct8.moduli)[:ct12_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct12_moduli_src = getattr(ct4, "moduli", getattr(ct8, "moduli", v0.q_towers))
  if isinstance(ct12_moduli_src, (int, np.integer)):
    ct12_moduli_src = [ct12_moduli_src]
  ct12_moduli = jnp.array(
      list(ct12_moduli_src)[:ct12_num_moduli], dtype=jnp.uint64
  )
  ct12_sum = ct12_lhs.astype(jnp.uint64) + ct12_rhs.astype(jnp.uint64)
  ct12 = jnp.where(
      ct12_sum >= ct12_moduli, ct12_sum - ct12_moduli, ct12_sum
  ).astype(jnp.uint32)
  ct13_lhs = ct12.polynomial if hasattr(ct12, "polynomial") else ct12
  ct13_rhs = ct9.polynomial if hasattr(ct9, "polynomial") else ct9
  ct13_lhs = ct13_lhs.reshape(
      ct13_lhs.shape[0],
      ct13_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct13_lhs.shape[-1],
  )
  ct13_rhs = ct13_rhs.reshape(
      ct13_rhs.shape[0],
      ct13_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct13_rhs.shape[-1],
  )
  if ct13_lhs.shape != ct13_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct13_num_moduli = ct13_lhs.shape[-1]
  if hasattr(ct12, "moduli") and hasattr(ct9, "moduli"):
    if (
        list(ct12.moduli)[:ct13_num_moduli]
        != list(ct9.moduli)[:ct13_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct13_moduli_src = getattr(ct12, "moduli", getattr(ct9, "moduli", v0.q_towers))
  if isinstance(ct13_moduli_src, (int, np.integer)):
    ct13_moduli_src = [ct13_moduli_src]
  ct13_moduli = jnp.array(
      list(ct13_moduli_src)[:ct13_num_moduli], dtype=jnp.uint64
  )
  ct13_sum = ct13_lhs.astype(jnp.uint64) + ct13_rhs.astype(jnp.uint64)
  ct13 = jnp.where(
      ct13_sum >= ct13_moduli, ct13_sum - ct13_moduli, ct13_sum
  ).astype(jnp.uint32)
  ct14_lhs = ct11.polynomial if hasattr(ct11, "polynomial") else ct11
  ct14_rhs = ct13.polynomial if hasattr(ct13, "polynomial") else ct13
  ct14_lhs = ct14_lhs.reshape(
      ct14_lhs.shape[0],
      ct14_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct14_lhs.shape[-1],
  )
  ct14_rhs = ct14_rhs.reshape(
      ct14_rhs.shape[0],
      ct14_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct14_rhs.shape[-1],
  )
  if ct14_lhs.shape != ct14_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct14_num_moduli = ct14_lhs.shape[-1]
  if hasattr(ct11, "moduli") and hasattr(ct13, "moduli"):
    if (
        list(ct11.moduli)[:ct14_num_moduli]
        != list(ct13.moduli)[:ct14_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct14_moduli_src = getattr(
      ct11, "moduli", getattr(ct13, "moduli", v0.q_towers)
  )
  if isinstance(ct14_moduli_src, (int, np.integer)):
    ct14_moduli_src = [ct14_moduli_src]
  ct14_moduli = jnp.array(
      list(ct14_moduli_src)[:ct14_num_moduli], dtype=jnp.uint64
  )
  ct14_sum = ct14_lhs.astype(jnp.uint64) + ct14_rhs.astype(jnp.uint64)
  ct14 = jnp.where(
      ct14_sum >= ct14_moduli, ct14_sum - ct14_moduli, ct14_sum
  ).astype(jnp.uint32)
  v10 = [None] * 1
  ct15_arg = _ensure_poly(v0, ct14, v0.max_level)
  ct15 = v0.he_rescale[v0.max_level, v0.max_level - 1](ct15_arg)
  v10[0] = ct15
  v11 = v10
  return v11


def matvec_identity(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  (v3, v4) = matvec_identity__preprocessing(v0, v1)
  v5 = matvec_identity__preprocessed(v0, v1, v2, v3, v4)
  return v5


def matvec_identity__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v15)
  v0.public_key = v3
  ct_raw = v0.encrypt(pt)
  ct = _ensure_poly(v0, ct_raw)
  v16 = [ct]
  return v16


def matvec_identity__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 7
  v8 = 0
  v9 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  pt_ct = _ensure_poly(v0, ct)
  _num_moduli = pt_ct.polynomial.shape[-1]
  _q_sub = list(getattr(pt_ct, "moduli", v0.q_towers))[:_num_moduli]
  _ct_for_dec = Polynomial(
      {
          "batch": pt_ct.polynomial.shape[0],
          "num_elements": pt_ct.polynomial.shape[1],
          "degree": v0.degree,
          "precision": 32,
          "num_moduli": _num_moduli,
          "degree_layout": (v0.degree,),
      },
      {"moduli": _q_sub},
  )
  _ct_for_dec.set_batch_polynomial(
      pt_ct.polynomial.reshape(
          pt_ct.polynomial.shape[0],
          pt_ct.polynomial.shape[1],
          v0.degree,
          _num_moduli,
      )
  )
  pt = v0.decrypt(_ct_for_dec)
  v10 = v0.decode(pt, is_ntt=False).real.reshape(1, 8)
  v11 = v9.copy()
  for v12 in range(0, 8):
    v14 = v7 - v12
    v15 = int(v14)
    v16 = v10[0, v15]
    v11[v15] = v16
  return v11


def matvec_shift__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> (np.ndarray, np.ndarray):
  v2 = np.full((8,), 0.000000e00, dtype=np.float32)
  v3 = np.full((8,), 1.000000e00, dtype=np.float32)
  pt = v0.encode(v2)
  pt1 = v0.encode(v3)
  v4 = [pt]
  v5 = [pt1]
  return (v4, v5)


def matvec_shift__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
) -> np.ndarray:
  v5 = 1
  v6 = 2
  v7 = 3
  v8 = 6
  v9 = 0
  pt = v3[0]
  pt1 = v4[0]
  ct = v2[0]
  ct1_arg = _ensure_poly(v0, ct, v0.max_level)
  ct1_pt_ntt = (
      pt.polynomial[0, 0, :, : ct1_arg.polynomial.shape[-1]]
      .reshape(ct1_arg.r, ct1_arg.c, ct1_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1 = ct1_ptct.mul(ct1_arg, use_bat=False)
  ct2_arg = _ensure_poly(v0, ct, v0.max_level)
  ct2 = v0.he_rot[v0.max_level, 1].rotate(ct2_arg)
  ct3_arg = _ensure_poly(v0, ct, v0.max_level)
  ct3 = v0.he_rot[v0.max_level, 2].rotate(ct3_arg)
  ct4_arg = _ensure_poly(v0, ct3, v0.max_level)
  ct4_pt_ntt = (
      pt.polynomial[0, 0, :, : ct4_arg.polynomial.shape[-1]]
      .reshape(ct4_arg.r, ct4_arg.c, ct4_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct4_ptct = v0.ptct_mul[v0.max_level]
  ct4_ptct.set_plaintext(ct4_pt_ntt)
  ct4 = ct4_ptct.mul(ct4_arg, use_bat=False)
  ct5_arg = _ensure_poly(v0, ct2, v0.max_level)
  ct5_pt_ntt = (
      pt.polynomial[0, 0, :, : ct5_arg.polynomial.shape[-1]]
      .reshape(ct5_arg.r, ct5_arg.c, ct5_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5 = ct5_ptct.mul(ct5_arg, use_bat=False)
  ct6_lhs = ct1.polynomial if hasattr(ct1, "polynomial") else ct1
  ct6_rhs = ct5.polynomial if hasattr(ct5, "polynomial") else ct5
  ct6_lhs = ct6_lhs.reshape(
      ct6_lhs.shape[0],
      ct6_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct6_lhs.shape[-1],
  )
  ct6_rhs = ct6_rhs.reshape(
      ct6_rhs.shape[0],
      ct6_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct6_rhs.shape[-1],
  )
  if ct6_lhs.shape != ct6_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct6_num_moduli = ct6_lhs.shape[-1]
  if hasattr(ct1, "moduli") and hasattr(ct5, "moduli"):
    if list(ct1.moduli)[:ct6_num_moduli] != list(ct5.moduli)[:ct6_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct6_moduli_src = getattr(ct1, "moduli", getattr(ct5, "moduli", v0.q_towers))
  if isinstance(ct6_moduli_src, (int, np.integer)):
    ct6_moduli_src = [ct6_moduli_src]
  ct6_moduli = jnp.array(
      list(ct6_moduli_src)[:ct6_num_moduli], dtype=jnp.uint64
  )
  ct6_sum = ct6_lhs.astype(jnp.uint64) + ct6_rhs.astype(jnp.uint64)
  ct6 = jnp.where(ct6_sum >= ct6_moduli, ct6_sum - ct6_moduli, ct6_sum).astype(
      jnp.uint32
  )
  ct7_lhs = ct6.polynomial if hasattr(ct6, "polynomial") else ct6
  ct7_rhs = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  ct7_lhs = ct7_lhs.reshape(
      ct7_lhs.shape[0],
      ct7_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct7_lhs.shape[-1],
  )
  ct7_rhs = ct7_rhs.reshape(
      ct7_rhs.shape[0],
      ct7_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct7_rhs.shape[-1],
  )
  if ct7_lhs.shape != ct7_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct7_num_moduli = ct7_lhs.shape[-1]
  if hasattr(ct6, "moduli") and hasattr(ct4, "moduli"):
    if list(ct6.moduli)[:ct7_num_moduli] != list(ct4.moduli)[:ct7_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct7_moduli_src = getattr(ct6, "moduli", getattr(ct4, "moduli", v0.q_towers))
  if isinstance(ct7_moduli_src, (int, np.integer)):
    ct7_moduli_src = [ct7_moduli_src]
  ct7_moduli = jnp.array(
      list(ct7_moduli_src)[:ct7_num_moduli], dtype=jnp.uint64
  )
  ct7_sum = ct7_lhs.astype(jnp.uint64) + ct7_rhs.astype(jnp.uint64)
  ct7 = jnp.where(ct7_sum >= ct7_moduli, ct7_sum - ct7_moduli, ct7_sum).astype(
      jnp.uint32
  )
  ct8_arg = _ensure_poly(v0, ct7, v0.max_level)
  ct8 = v0.he_rot[v0.max_level, 3].rotate(ct8_arg)
  ct9_arg = _ensure_poly(v0, ct6, v0.max_level)
  ct9 = v0.he_rot[v0.max_level, 6].rotate(ct9_arg)
  ct10_arg = _ensure_poly(v0, ct2, v0.max_level)
  ct10_pt_ntt = (
      pt1.polynomial[0, 0, :, : ct10_arg.polynomial.shape[-1]]
      .reshape(ct10_arg.r, ct10_arg.c, ct10_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct10_ptct = v0.ptct_mul[v0.max_level]
  ct10_ptct.set_plaintext(ct10_pt_ntt)
  ct10 = ct10_ptct.mul(ct10_arg, use_bat=False)
  ct11_lhs = ct1.polynomial if hasattr(ct1, "polynomial") else ct1
  ct11_rhs = ct10.polynomial if hasattr(ct10, "polynomial") else ct10
  ct11_lhs = ct11_lhs.reshape(
      ct11_lhs.shape[0],
      ct11_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct11_lhs.shape[-1],
  )
  ct11_rhs = ct11_rhs.reshape(
      ct11_rhs.shape[0],
      ct11_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct11_rhs.shape[-1],
  )
  if ct11_lhs.shape != ct11_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct11_num_moduli = ct11_lhs.shape[-1]
  if hasattr(ct1, "moduli") and hasattr(ct10, "moduli"):
    if (
        list(ct1.moduli)[:ct11_num_moduli]
        != list(ct10.moduli)[:ct11_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct11_moduli_src = getattr(ct1, "moduli", getattr(ct10, "moduli", v0.q_towers))
  if isinstance(ct11_moduli_src, (int, np.integer)):
    ct11_moduli_src = [ct11_moduli_src]
  ct11_moduli = jnp.array(
      list(ct11_moduli_src)[:ct11_num_moduli], dtype=jnp.uint64
  )
  ct11_sum = ct11_lhs.astype(jnp.uint64) + ct11_rhs.astype(jnp.uint64)
  ct11 = jnp.where(
      ct11_sum >= ct11_moduli, ct11_sum - ct11_moduli, ct11_sum
  ).astype(jnp.uint32)
  ct12_lhs = ct4.polynomial if hasattr(ct4, "polynomial") else ct4
  ct12_rhs = ct8.polynomial if hasattr(ct8, "polynomial") else ct8
  ct12_lhs = ct12_lhs.reshape(
      ct12_lhs.shape[0],
      ct12_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct12_lhs.shape[-1],
  )
  ct12_rhs = ct12_rhs.reshape(
      ct12_rhs.shape[0],
      ct12_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct12_rhs.shape[-1],
  )
  if ct12_lhs.shape != ct12_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct12_num_moduli = ct12_lhs.shape[-1]
  if hasattr(ct4, "moduli") and hasattr(ct8, "moduli"):
    if list(ct4.moduli)[:ct12_num_moduli] != list(ct8.moduli)[:ct12_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct12_moduli_src = getattr(ct4, "moduli", getattr(ct8, "moduli", v0.q_towers))
  if isinstance(ct12_moduli_src, (int, np.integer)):
    ct12_moduli_src = [ct12_moduli_src]
  ct12_moduli = jnp.array(
      list(ct12_moduli_src)[:ct12_num_moduli], dtype=jnp.uint64
  )
  ct12_sum = ct12_lhs.astype(jnp.uint64) + ct12_rhs.astype(jnp.uint64)
  ct12 = jnp.where(
      ct12_sum >= ct12_moduli, ct12_sum - ct12_moduli, ct12_sum
  ).astype(jnp.uint32)
  ct13_lhs = ct12.polynomial if hasattr(ct12, "polynomial") else ct12
  ct13_rhs = ct9.polynomial if hasattr(ct9, "polynomial") else ct9
  ct13_lhs = ct13_lhs.reshape(
      ct13_lhs.shape[0],
      ct13_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct13_lhs.shape[-1],
  )
  ct13_rhs = ct13_rhs.reshape(
      ct13_rhs.shape[0],
      ct13_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct13_rhs.shape[-1],
  )
  if ct13_lhs.shape != ct13_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct13_num_moduli = ct13_lhs.shape[-1]
  if hasattr(ct12, "moduli") and hasattr(ct9, "moduli"):
    if (
        list(ct12.moduli)[:ct13_num_moduli]
        != list(ct9.moduli)[:ct13_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct13_moduli_src = getattr(ct12, "moduli", getattr(ct9, "moduli", v0.q_towers))
  if isinstance(ct13_moduli_src, (int, np.integer)):
    ct13_moduli_src = [ct13_moduli_src]
  ct13_moduli = jnp.array(
      list(ct13_moduli_src)[:ct13_num_moduli], dtype=jnp.uint64
  )
  ct13_sum = ct13_lhs.astype(jnp.uint64) + ct13_rhs.astype(jnp.uint64)
  ct13 = jnp.where(
      ct13_sum >= ct13_moduli, ct13_sum - ct13_moduli, ct13_sum
  ).astype(jnp.uint32)
  ct14_lhs = ct11.polynomial if hasattr(ct11, "polynomial") else ct11
  ct14_rhs = ct13.polynomial if hasattr(ct13, "polynomial") else ct13
  ct14_lhs = ct14_lhs.reshape(
      ct14_lhs.shape[0],
      ct14_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct14_lhs.shape[-1],
  )
  ct14_rhs = ct14_rhs.reshape(
      ct14_rhs.shape[0],
      ct14_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct14_rhs.shape[-1],
  )
  if ct14_lhs.shape != ct14_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct14_num_moduli = ct14_lhs.shape[-1]
  if hasattr(ct11, "moduli") and hasattr(ct13, "moduli"):
    if (
        list(ct11.moduli)[:ct14_num_moduli]
        != list(ct13.moduli)[:ct14_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct14_moduli_src = getattr(
      ct11, "moduli", getattr(ct13, "moduli", v0.q_towers)
  )
  if isinstance(ct14_moduli_src, (int, np.integer)):
    ct14_moduli_src = [ct14_moduli_src]
  ct14_moduli = jnp.array(
      list(ct14_moduli_src)[:ct14_num_moduli], dtype=jnp.uint64
  )
  ct14_sum = ct14_lhs.astype(jnp.uint64) + ct14_rhs.astype(jnp.uint64)
  ct14 = jnp.where(
      ct14_sum >= ct14_moduli, ct14_sum - ct14_moduli, ct14_sum
  ).astype(jnp.uint32)
  v10 = [None] * 1
  ct15_arg = _ensure_poly(v0, ct14, v0.max_level)
  ct15 = v0.he_rescale[v0.max_level, v0.max_level - 1](ct15_arg)
  v10[0] = ct15
  v11 = v10
  return v11


def matvec_shift(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  (v3, v4) = matvec_shift__preprocessing(v0, v1)
  v5 = matvec_shift__preprocessed(v0, v1, v2, v3, v4)
  return v5


def matvec_shift__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v15)
  v0.public_key = v3
  ct_raw = v0.encrypt(pt)
  ct = _ensure_poly(v0, ct_raw)
  v16 = [ct]
  return v16


def matvec_shift__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 7
  v8 = 0
  v9 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  pt_ct = _ensure_poly(v0, ct)
  _num_moduli = pt_ct.polynomial.shape[-1]
  _q_sub = list(getattr(pt_ct, "moduli", v0.q_towers))[:_num_moduli]
  _ct_for_dec = Polynomial(
      {
          "batch": pt_ct.polynomial.shape[0],
          "num_elements": pt_ct.polynomial.shape[1],
          "degree": v0.degree,
          "precision": 32,
          "num_moduli": _num_moduli,
          "degree_layout": (v0.degree,),
      },
      {"moduli": _q_sub},
  )
  _ct_for_dec.set_batch_polynomial(
      pt_ct.polynomial.reshape(
          pt_ct.polynomial.shape[0],
          pt_ct.polynomial.shape[1],
          v0.degree,
          _num_moduli,
      )
  )
  pt = v0.decrypt(_ct_for_dec)
  v10 = v0.decode(pt, is_ntt=False).real.reshape(1, 8)
  v11 = v9.copy()
  for v12 in range(0, 8):
    v14 = v7 - v12
    v15 = int(v14)
    v16 = v10[0, v15]
    v11[v15] = v16
  return v11


def matvec_random__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> (
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
):
  v2 = np.array(
      [
          8.116263e-01,
          1.445338e00,
          9.206955e-01,
          1.077045e00,
          6.787661e-01,
          1.358792e00,
          1.236010e00,
          7.778313e-01,
      ],
      dtype=np.float32,
  )
  v3 = np.array(
      [
          1.906357e00,
          1.391105e-01,
          6.533354e-01,
          1.225588e00,
          2.855770e-01,
          6.922510e-01,
          1.851561e00,
          2.681358e-01,
      ],
      dtype=np.float32,
  )
  v4 = np.array(
      [
          1.490788e00,
          1.942829e00,
          1.262521e00,
          1.882558e-01,
          1.400043e00,
          1.088129e00,
          1.138749e00,
          4.723674e-01,
      ],
      dtype=np.float32,
  )
  v5 = np.array(
      [
          3.318726e-01,
          4.512235e-01,
          1.859318e-01,
          1.237451e00,
          1.681641e00,
          3.650383e-01,
          1.254335e00,
          9.362897e-01,
      ],
      dtype=np.float32,
  )
  v6 = np.array(
      [
          1.040836e00,
          1.942211e00,
          7.181276e-01,
          3.964354e-01,
          5.034443e-01,
          6.550748e-01,
          4.239958e-01,
          2.235980e-01,
      ],
      dtype=np.float32,
  )
  v7 = np.array(
      [
          1.653382e-01,
          1.572752e00,
          8.384869e-01,
          3.963896e-01,
          4.454674e-01,
          7.960875e-01,
          9.665329e-01,
          1.902883e00,
      ],
      dtype=np.float32,
  )
  v8 = np.array(
      [
          6.780602e-01,
          1.591834e00,
          1.934701e00,
          1.827709e00,
          1.885048e00,
          6.155632e-01,
          2.103589e-01,
          4.484686e-01,
      ],
      dtype=np.float32,
  )
  v9 = np.array(
      [
          1.097037e00,
          4.793802e-01,
          1.635955e00,
          5.916820e-01,
          1.800172e00,
          1.674601e00,
          1.745735e00,
          1.242118e00,
      ],
      dtype=np.float32,
  )
  pt = v0.encode(v2)
  pt1 = v0.encode(v3)
  pt2 = v0.encode(v4)
  pt3 = v0.encode(v5)
  pt4 = v0.encode(v6)
  pt5 = v0.encode(v7)
  pt6 = v0.encode(v8)
  pt7 = v0.encode(v9)
  v10 = [pt]
  v11 = [pt1]
  v12 = [pt2]
  v13 = [pt3]
  v14 = [pt4]
  v15 = [pt5]
  v16 = [pt6]
  v17 = [pt7]
  return (v10, v11, v12, v13, v14, v15, v16, v17)


def matvec_random__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
    v5: np.ndarray,
    v6: np.ndarray,
    v7: np.ndarray,
    v8: np.ndarray,
    v9: np.ndarray,
    v10: np.ndarray,
) -> np.ndarray:
  v11 = 1
  v12 = 2
  v13 = 3
  v14 = 6
  v15 = 0
  pt = v3[0]
  pt1 = v4[0]
  pt2 = v5[0]
  pt3 = v6[0]
  pt4 = v7[0]
  pt5 = v8[0]
  pt6 = v9[0]
  pt7 = v10[0]
  ct = v2[0]
  ct1_arg = _ensure_poly(v0, ct, v0.max_level)
  ct1_pt_ntt = (
      pt.polynomial[0, 0, :, : ct1_arg.polynomial.shape[-1]]
      .reshape(ct1_arg.r, ct1_arg.c, ct1_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1 = ct1_ptct.mul(ct1_arg, use_bat=False)
  ct2_arg = _ensure_poly(v0, ct, v0.max_level)
  ct2 = v0.he_rot[v0.max_level, 1].rotate(ct2_arg)
  ct3_arg = _ensure_poly(v0, ct2, v0.max_level)
  ct3_pt_ntt = (
      pt1.polynomial[0, 0, :, : ct3_arg.polynomial.shape[-1]]
      .reshape(ct3_arg.r, ct3_arg.c, ct3_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct3_ptct = v0.ptct_mul[v0.max_level]
  ct3_ptct.set_plaintext(ct3_pt_ntt)
  ct3 = ct3_ptct.mul(ct3_arg, use_bat=False)
  ct4_arg = _ensure_poly(v0, ct, v0.max_level)
  ct4 = v0.he_rot[v0.max_level, 2].rotate(ct4_arg)
  ct5_arg = _ensure_poly(v0, ct4, v0.max_level)
  ct5_pt_ntt = (
      pt2.polynomial[0, 0, :, : ct5_arg.polynomial.shape[-1]]
      .reshape(ct5_arg.r, ct5_arg.c, ct5_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5 = ct5_ptct.mul(ct5_arg, use_bat=False)
  ct6_arg = _ensure_poly(v0, ct, v0.max_level)
  ct6_pt_ntt = (
      pt3.polynomial[0, 0, :, : ct6_arg.polynomial.shape[-1]]
      .reshape(ct6_arg.r, ct6_arg.c, ct6_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct6_ptct = v0.ptct_mul[v0.max_level]
  ct6_ptct.set_plaintext(ct6_pt_ntt)
  ct6 = ct6_ptct.mul(ct6_arg, use_bat=False)
  ct7_arg = _ensure_poly(v0, ct2, v0.max_level)
  ct7_pt_ntt = (
      pt4.polynomial[0, 0, :, : ct7_arg.polynomial.shape[-1]]
      .reshape(ct7_arg.r, ct7_arg.c, ct7_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct7_ptct = v0.ptct_mul[v0.max_level]
  ct7_ptct.set_plaintext(ct7_pt_ntt)
  ct7 = ct7_ptct.mul(ct7_arg, use_bat=False)
  ct8_arg = _ensure_poly(v0, ct4, v0.max_level)
  ct8_pt_ntt = (
      pt5.polynomial[0, 0, :, : ct8_arg.polynomial.shape[-1]]
      .reshape(ct8_arg.r, ct8_arg.c, ct8_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct8_ptct = v0.ptct_mul[v0.max_level]
  ct8_ptct.set_plaintext(ct8_pt_ntt)
  ct8 = ct8_ptct.mul(ct8_arg, use_bat=False)
  ct9_lhs = ct6.polynomial if hasattr(ct6, "polynomial") else ct6
  ct9_rhs = ct7.polynomial if hasattr(ct7, "polynomial") else ct7
  ct9_lhs = ct9_lhs.reshape(
      ct9_lhs.shape[0],
      ct9_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct9_lhs.shape[-1],
  )
  ct9_rhs = ct9_rhs.reshape(
      ct9_rhs.shape[0],
      ct9_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct9_rhs.shape[-1],
  )
  if ct9_lhs.shape != ct9_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct9_num_moduli = ct9_lhs.shape[-1]
  if hasattr(ct6, "moduli") and hasattr(ct7, "moduli"):
    if list(ct6.moduli)[:ct9_num_moduli] != list(ct7.moduli)[:ct9_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct9_moduli_src = getattr(ct6, "moduli", getattr(ct7, "moduli", v0.q_towers))
  if isinstance(ct9_moduli_src, (int, np.integer)):
    ct9_moduli_src = [ct9_moduli_src]
  ct9_moduli = jnp.array(
      list(ct9_moduli_src)[:ct9_num_moduli], dtype=jnp.uint64
  )
  ct9_sum = ct9_lhs.astype(jnp.uint64) + ct9_rhs.astype(jnp.uint64)
  ct9 = jnp.where(ct9_sum >= ct9_moduli, ct9_sum - ct9_moduli, ct9_sum).astype(
      jnp.uint32
  )
  ct10_lhs = ct9.polynomial if hasattr(ct9, "polynomial") else ct9
  ct10_rhs = ct8.polynomial if hasattr(ct8, "polynomial") else ct8
  ct10_lhs = ct10_lhs.reshape(
      ct10_lhs.shape[0],
      ct10_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct10_lhs.shape[-1],
  )
  ct10_rhs = ct10_rhs.reshape(
      ct10_rhs.shape[0],
      ct10_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct10_rhs.shape[-1],
  )
  if ct10_lhs.shape != ct10_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct10_num_moduli = ct10_lhs.shape[-1]
  if hasattr(ct9, "moduli") and hasattr(ct8, "moduli"):
    if list(ct9.moduli)[:ct10_num_moduli] != list(ct8.moduli)[:ct10_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct10_moduli_src = getattr(ct9, "moduli", getattr(ct8, "moduli", v0.q_towers))
  if isinstance(ct10_moduli_src, (int, np.integer)):
    ct10_moduli_src = [ct10_moduli_src]
  ct10_moduli = jnp.array(
      list(ct10_moduli_src)[:ct10_num_moduli], dtype=jnp.uint64
  )
  ct10_sum = ct10_lhs.astype(jnp.uint64) + ct10_rhs.astype(jnp.uint64)
  ct10 = jnp.where(
      ct10_sum >= ct10_moduli, ct10_sum - ct10_moduli, ct10_sum
  ).astype(jnp.uint32)
  ct11_arg = _ensure_poly(v0, ct10, v0.max_level)
  ct11 = v0.he_rot[v0.max_level, 3].rotate(ct11_arg)
  ct12_arg = _ensure_poly(v0, ct, v0.max_level)
  ct12_pt_ntt = (
      pt6.polynomial[0, 0, :, : ct12_arg.polynomial.shape[-1]]
      .reshape(ct12_arg.r, ct12_arg.c, ct12_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct12_ptct = v0.ptct_mul[v0.max_level]
  ct12_ptct.set_plaintext(ct12_pt_ntt)
  ct12 = ct12_ptct.mul(ct12_arg, use_bat=False)
  ct13_arg = _ensure_poly(v0, ct2, v0.max_level)
  ct13_pt_ntt = (
      pt7.polynomial[0, 0, :, : ct13_arg.polynomial.shape[-1]]
      .reshape(ct13_arg.r, ct13_arg.c, ct13_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct13_ptct = v0.ptct_mul[v0.max_level]
  ct13_ptct.set_plaintext(ct13_pt_ntt)
  ct13 = ct13_ptct.mul(ct13_arg, use_bat=False)
  ct14_lhs = ct12.polynomial if hasattr(ct12, "polynomial") else ct12
  ct14_rhs = ct13.polynomial if hasattr(ct13, "polynomial") else ct13
  ct14_lhs = ct14_lhs.reshape(
      ct14_lhs.shape[0],
      ct14_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct14_lhs.shape[-1],
  )
  ct14_rhs = ct14_rhs.reshape(
      ct14_rhs.shape[0],
      ct14_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct14_rhs.shape[-1],
  )
  if ct14_lhs.shape != ct14_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct14_num_moduli = ct14_lhs.shape[-1]
  if hasattr(ct12, "moduli") and hasattr(ct13, "moduli"):
    if (
        list(ct12.moduli)[:ct14_num_moduli]
        != list(ct13.moduli)[:ct14_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct14_moduli_src = getattr(
      ct12, "moduli", getattr(ct13, "moduli", v0.q_towers)
  )
  if isinstance(ct14_moduli_src, (int, np.integer)):
    ct14_moduli_src = [ct14_moduli_src]
  ct14_moduli = jnp.array(
      list(ct14_moduli_src)[:ct14_num_moduli], dtype=jnp.uint64
  )
  ct14_sum = ct14_lhs.astype(jnp.uint64) + ct14_rhs.astype(jnp.uint64)
  ct14 = jnp.where(
      ct14_sum >= ct14_moduli, ct14_sum - ct14_moduli, ct14_sum
  ).astype(jnp.uint32)
  ct15_arg = _ensure_poly(v0, ct14, v0.max_level)
  ct15 = v0.he_rot[v0.max_level, 6].rotate(ct15_arg)
  ct16_lhs = ct1.polynomial if hasattr(ct1, "polynomial") else ct1
  ct16_rhs = ct3.polynomial if hasattr(ct3, "polynomial") else ct3
  ct16_lhs = ct16_lhs.reshape(
      ct16_lhs.shape[0],
      ct16_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct16_lhs.shape[-1],
  )
  ct16_rhs = ct16_rhs.reshape(
      ct16_rhs.shape[0],
      ct16_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct16_rhs.shape[-1],
  )
  if ct16_lhs.shape != ct16_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct16_num_moduli = ct16_lhs.shape[-1]
  if hasattr(ct1, "moduli") and hasattr(ct3, "moduli"):
    if list(ct1.moduli)[:ct16_num_moduli] != list(ct3.moduli)[:ct16_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct16_moduli_src = getattr(ct1, "moduli", getattr(ct3, "moduli", v0.q_towers))
  if isinstance(ct16_moduli_src, (int, np.integer)):
    ct16_moduli_src = [ct16_moduli_src]
  ct16_moduli = jnp.array(
      list(ct16_moduli_src)[:ct16_num_moduli], dtype=jnp.uint64
  )
  ct16_sum = ct16_lhs.astype(jnp.uint64) + ct16_rhs.astype(jnp.uint64)
  ct16 = jnp.where(
      ct16_sum >= ct16_moduli, ct16_sum - ct16_moduli, ct16_sum
  ).astype(jnp.uint32)
  ct17_lhs = ct5.polynomial if hasattr(ct5, "polynomial") else ct5
  ct17_rhs = ct11.polynomial if hasattr(ct11, "polynomial") else ct11
  ct17_lhs = ct17_lhs.reshape(
      ct17_lhs.shape[0],
      ct17_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct17_lhs.shape[-1],
  )
  ct17_rhs = ct17_rhs.reshape(
      ct17_rhs.shape[0],
      ct17_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct17_rhs.shape[-1],
  )
  if ct17_lhs.shape != ct17_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct17_num_moduli = ct17_lhs.shape[-1]
  if hasattr(ct5, "moduli") and hasattr(ct11, "moduli"):
    if (
        list(ct5.moduli)[:ct17_num_moduli]
        != list(ct11.moduli)[:ct17_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct17_moduli_src = getattr(ct5, "moduli", getattr(ct11, "moduli", v0.q_towers))
  if isinstance(ct17_moduli_src, (int, np.integer)):
    ct17_moduli_src = [ct17_moduli_src]
  ct17_moduli = jnp.array(
      list(ct17_moduli_src)[:ct17_num_moduli], dtype=jnp.uint64
  )
  ct17_sum = ct17_lhs.astype(jnp.uint64) + ct17_rhs.astype(jnp.uint64)
  ct17 = jnp.where(
      ct17_sum >= ct17_moduli, ct17_sum - ct17_moduli, ct17_sum
  ).astype(jnp.uint32)
  ct18_lhs = ct17.polynomial if hasattr(ct17, "polynomial") else ct17
  ct18_rhs = ct15.polynomial if hasattr(ct15, "polynomial") else ct15
  ct18_lhs = ct18_lhs.reshape(
      ct18_lhs.shape[0],
      ct18_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct18_lhs.shape[-1],
  )
  ct18_rhs = ct18_rhs.reshape(
      ct18_rhs.shape[0],
      ct18_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct18_rhs.shape[-1],
  )
  if ct18_lhs.shape != ct18_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct18_num_moduli = ct18_lhs.shape[-1]
  if hasattr(ct17, "moduli") and hasattr(ct15, "moduli"):
    if (
        list(ct17.moduli)[:ct18_num_moduli]
        != list(ct15.moduli)[:ct18_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct18_moduli_src = getattr(
      ct17, "moduli", getattr(ct15, "moduli", v0.q_towers)
  )
  if isinstance(ct18_moduli_src, (int, np.integer)):
    ct18_moduli_src = [ct18_moduli_src]
  ct18_moduli = jnp.array(
      list(ct18_moduli_src)[:ct18_num_moduli], dtype=jnp.uint64
  )
  ct18_sum = ct18_lhs.astype(jnp.uint64) + ct18_rhs.astype(jnp.uint64)
  ct18 = jnp.where(
      ct18_sum >= ct18_moduli, ct18_sum - ct18_moduli, ct18_sum
  ).astype(jnp.uint32)
  ct19_lhs = ct16.polynomial if hasattr(ct16, "polynomial") else ct16
  ct19_rhs = ct18.polynomial if hasattr(ct18, "polynomial") else ct18
  ct19_lhs = ct19_lhs.reshape(
      ct19_lhs.shape[0],
      ct19_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct19_lhs.shape[-1],
  )
  ct19_rhs = ct19_rhs.reshape(
      ct19_rhs.shape[0],
      ct19_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct19_rhs.shape[-1],
  )
  if ct19_lhs.shape != ct19_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct19_num_moduli = ct19_lhs.shape[-1]
  if hasattr(ct16, "moduli") and hasattr(ct18, "moduli"):
    if (
        list(ct16.moduli)[:ct19_num_moduli]
        != list(ct18.moduli)[:ct19_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct19_moduli_src = getattr(
      ct16, "moduli", getattr(ct18, "moduli", v0.q_towers)
  )
  if isinstance(ct19_moduli_src, (int, np.integer)):
    ct19_moduli_src = [ct19_moduli_src]
  ct19_moduli = jnp.array(
      list(ct19_moduli_src)[:ct19_num_moduli], dtype=jnp.uint64
  )
  ct19_sum = ct19_lhs.astype(jnp.uint64) + ct19_rhs.astype(jnp.uint64)
  ct19 = jnp.where(
      ct19_sum >= ct19_moduli, ct19_sum - ct19_moduli, ct19_sum
  ).astype(jnp.uint32)
  v16 = [None] * 1
  ct20_arg = _ensure_poly(v0, ct19, v0.max_level)
  ct20 = v0.he_rescale[v0.max_level, v0.max_level - 1](ct20_arg)
  v16[0] = ct20
  v17 = v16
  return v17


def matvec_random(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  (v3, v4, v5, v6, v7, v8, v9, v10) = matvec_random__preprocessing(v0, v1)
  v11 = matvec_random__preprocessed(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)
  return v11


def matvec_random__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v15)
  v0.public_key = v3
  ct_raw = v0.encrypt(pt)
  ct = _ensure_poly(v0, ct_raw)
  v16 = [ct]
  return v16


def matvec_random__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 7
  v8 = 0
  v9 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  pt_ct = _ensure_poly(v0, ct)
  _num_moduli = pt_ct.polynomial.shape[-1]
  _q_sub = list(getattr(pt_ct, "moduli", v0.q_towers))[:_num_moduli]
  _ct_for_dec = Polynomial(
      {
          "batch": pt_ct.polynomial.shape[0],
          "num_elements": pt_ct.polynomial.shape[1],
          "degree": v0.degree,
          "precision": 32,
          "num_moduli": _num_moduli,
          "degree_layout": (v0.degree,),
      },
      {"moduli": _q_sub},
  )
  _ct_for_dec.set_batch_polynomial(
      pt_ct.polynomial.reshape(
          pt_ct.polynomial.shape[0],
          pt_ct.polynomial.shape[1],
          v0.degree,
          _num_moduli,
      )
  )
  pt = v0.decrypt(_ct_for_dec)
  v10 = v0.decode(pt, is_ntt=False).real.reshape(1, 8)
  v11 = v9.copy()
  for v12 in range(0, 8):
    v14 = v7 - v12
    v15 = int(v14)
    v16 = v10[0, v15]
    v11[v15] = v16
  return v11


def matvec_chain__preprocessing(
    v0: ckks.CKKSContext,
    v1: dict,
) -> (
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
):
  v2 = np.array(
      [
          1.340000e00,
          1.220000e00,
          1.050000e00,
          1.500000e00,
          1.010000e00,
          8.800000e-01,
          5.000000e-01,
          1.060000e00,
      ],
      dtype=np.float32,
  )
  v3 = np.array(
      [
          5.800000e-01,
          5.200000e-01,
          8.900000e-01,
          8.600000e-01,
          1.170000e00,
          8.200000e-01,
          1.490000e00,
          1.410000e00,
      ],
      dtype=np.float32,
  )
  v4 = np.array(
      [
          1.260000e00,
          1.090000e00,
          1.430000e00,
          1.260000e00,
          6.100000e-01,
          8.500000e-01,
          6.700000e-01,
          7.100000e-01,
      ],
      dtype=np.float32,
  )
  v5 = np.array(
      [
          8.200000e-01,
          1.330000e00,
          7.900000e-01,
          7.400000e-01,
          1.060000e00,
          1.340000e00,
          1.090000e00,
          6.300000e-01,
      ],
      dtype=np.float32,
  )
  v6 = np.array(
      [
          1.160000e00,
          8.400000e-01,
          1.020000e00,
          6.900000e-01,
          6.600000e-01,
          8.600000e-01,
          1.190000e00,
          6.500000e-01,
      ],
      dtype=np.float32,
  )
  v7 = np.array(
      [
          1.350000e00,
          1.050000e00,
          1.400000e00,
          1.070000e00,
          6.500000e-01,
          5.400000e-01,
          8.000000e-01,
          9.000000e-01,
      ],
      dtype=np.float32,
  )
  v8 = np.array(
      [
          8.200000e-01,
          9.000000e-01,
          7.400000e-01,
          1.050000e00,
          1.080000e00,
          1.480000e00,
          6.000000e-01,
          1.200000e00,
      ],
      dtype=np.float32,
  )
  v9 = np.array(
      [
          1.190000e00,
          1.200000e00,
          8.400000e-01,
          1.350000e00,
          1.020000e00,
          7.600000e-01,
          1.390000e00,
          1.130000e00,
      ],
      dtype=np.float32,
  )
  v10 = np.array(
      [
          1.200000e00,
          8.900000e-01,
          1.030000e00,
          7.300000e-01,
          9.300000e-01,
          7.500000e-01,
          8.400000e-01,
          1.170000e00,
      ],
      dtype=np.float32,
  )
  v11 = np.array(
      [
          7.900000e-01,
          8.400000e-01,
          1.030000e00,
          7.900000e-01,
          1.390000e00,
          9.800000e-01,
          8.000000e-01,
          9.200000e-01,
      ],
      dtype=np.float32,
  )
  v12 = np.array(
      [
          7.300000e-01,
          1.230000e00,
          1.130000e00,
          1.130000e00,
          1.440000e00,
          1.490000e00,
          1.020000e00,
          1.180000e00,
      ],
      dtype=np.float32,
  )
  v13 = np.array(
      [
          1.120000e00,
          1.110000e00,
          1.380000e00,
          1.050000e00,
          9.400000e-01,
          1.350000e00,
          5.900000e-01,
          1.000000e00,
      ],
      dtype=np.float32,
  )
  v14 = np.array(
      [
          6.200000e-01,
          6.200000e-01,
          1.010000e00,
          1.220000e00,
          5.600000e-01,
          1.220000e00,
          9.300000e-01,
          9.300000e-01,
      ],
      dtype=np.float32,
  )
  v15 = np.array(
      [
          8.200000e-01,
          1.330000e00,
          1.170000e00,
          9.200000e-01,
          9.000000e-01,
          1.110000e00,
          1.220000e00,
          9.900000e-01,
      ],
      dtype=np.float32,
  )
  v16 = np.array(
      [
          6.800000e-01,
          8.200000e-01,
          9.300000e-01,
          9.100000e-01,
          1.100000e00,
          1.090000e00,
          1.480000e00,
          1.240000e00,
      ],
      dtype=np.float32,
  )
  v17 = np.array(
      [
          6.800000e-01,
          8.600000e-01,
          8.100000e-01,
          1.370000e00,
          1.050000e00,
          1.120000e00,
          1.180000e00,
          9.800000e-01,
      ],
      dtype=np.float32,
  )
  pt = v0.encode(v2)
  pt1 = v0.encode(v3)
  pt2 = v0.encode(v4)
  pt3 = v0.encode(v5)
  pt4 = v0.encode(v6)
  pt5 = v0.encode(v7)
  pt6 = v0.encode(v8)
  pt7 = v0.encode(v9)
  pt8 = v0.encode(v10)
  pt9 = v0.encode(v11)
  pt10 = v0.encode(v12)
  pt11 = v0.encode(v13)
  pt12 = v0.encode(v14)
  pt13 = v0.encode(v15)
  pt14 = v0.encode(v16)
  pt15 = v0.encode(v17)
  v18 = [pt]
  v19 = [pt1]
  v20 = [pt2]
  v21 = [pt3]
  v22 = [pt4]
  v23 = [pt5]
  v24 = [pt6]
  v25 = [pt7]
  v26 = [pt8, pt9]
  v27 = [pt10, pt11]
  v28 = [pt12, pt13]
  v29 = [pt14, pt15]
  return (v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29)


def matvec_chain__preprocessed(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
    v4: np.ndarray,
    v5: np.ndarray,
    v6: np.ndarray,
    v7: np.ndarray,
    v8: np.ndarray,
    v9: np.ndarray,
    v10: np.ndarray,
    v11: np.ndarray,
    v12: np.ndarray,
    v13: np.ndarray,
    v14: np.ndarray,
) -> np.ndarray:
  v15 = 1
  v16 = 2
  v17 = 3
  v18 = 6
  v19 = 0
  pt = v3[0]
  pt1 = v4[0]
  pt2 = v5[0]
  pt3 = v6[0]
  pt4 = v7[0]
  pt5 = v8[0]
  pt6 = v9[0]
  pt7 = v10[0]
  pt8 = v11[0]
  pt9 = v11[1]
  pt10 = v12[0]
  pt11 = v12[1]
  pt12 = v13[0]
  pt13 = v13[1]
  pt14 = v14[0]
  pt15 = v14[1]
  ct = v2[0]
  ct1_arg = _ensure_poly(v0, ct, v0.max_level)
  ct1_pt_ntt = (
      pt.polynomial[0, 0, :, : ct1_arg.polynomial.shape[-1]]
      .reshape(ct1_arg.r, ct1_arg.c, ct1_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct1_ptct = v0.ptct_mul[v0.max_level]
  ct1_ptct.set_plaintext(ct1_pt_ntt)
  ct1 = ct1_ptct.mul(ct1_arg, use_bat=False)
  ct2_arg = _ensure_poly(v0, ct, v0.max_level)
  ct2 = v0.he_rot[v0.max_level, 1].rotate(ct2_arg)
  ct3_arg = _ensure_poly(v0, ct2, v0.max_level)
  ct3_pt_ntt = (
      pt1.polynomial[0, 0, :, : ct3_arg.polynomial.shape[-1]]
      .reshape(ct3_arg.r, ct3_arg.c, ct3_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct3_ptct = v0.ptct_mul[v0.max_level]
  ct3_ptct.set_plaintext(ct3_pt_ntt)
  ct3 = ct3_ptct.mul(ct3_arg, use_bat=False)
  ct4_arg = _ensure_poly(v0, ct, v0.max_level)
  ct4 = v0.he_rot[v0.max_level, 2].rotate(ct4_arg)
  ct5_arg = _ensure_poly(v0, ct4, v0.max_level)
  ct5_pt_ntt = (
      pt2.polynomial[0, 0, :, : ct5_arg.polynomial.shape[-1]]
      .reshape(ct5_arg.r, ct5_arg.c, ct5_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct5_ptct = v0.ptct_mul[v0.max_level]
  ct5_ptct.set_plaintext(ct5_pt_ntt)
  ct5 = ct5_ptct.mul(ct5_arg, use_bat=False)
  ct6_arg = _ensure_poly(v0, ct, v0.max_level)
  ct6_pt_ntt = (
      pt3.polynomial[0, 0, :, : ct6_arg.polynomial.shape[-1]]
      .reshape(ct6_arg.r, ct6_arg.c, ct6_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct6_ptct = v0.ptct_mul[v0.max_level]
  ct6_ptct.set_plaintext(ct6_pt_ntt)
  ct6 = ct6_ptct.mul(ct6_arg, use_bat=False)
  ct7_arg = _ensure_poly(v0, ct2, v0.max_level)
  ct7_pt_ntt = (
      pt4.polynomial[0, 0, :, : ct7_arg.polynomial.shape[-1]]
      .reshape(ct7_arg.r, ct7_arg.c, ct7_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct7_ptct = v0.ptct_mul[v0.max_level]
  ct7_ptct.set_plaintext(ct7_pt_ntt)
  ct7 = ct7_ptct.mul(ct7_arg, use_bat=False)
  ct8_arg = _ensure_poly(v0, ct4, v0.max_level)
  ct8_pt_ntt = (
      pt5.polynomial[0, 0, :, : ct8_arg.polynomial.shape[-1]]
      .reshape(ct8_arg.r, ct8_arg.c, ct8_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct8_ptct = v0.ptct_mul[v0.max_level]
  ct8_ptct.set_plaintext(ct8_pt_ntt)
  ct8 = ct8_ptct.mul(ct8_arg, use_bat=False)
  ct9_lhs = ct6.polynomial if hasattr(ct6, "polynomial") else ct6
  ct9_rhs = ct7.polynomial if hasattr(ct7, "polynomial") else ct7
  ct9_lhs = ct9_lhs.reshape(
      ct9_lhs.shape[0],
      ct9_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct9_lhs.shape[-1],
  )
  ct9_rhs = ct9_rhs.reshape(
      ct9_rhs.shape[0],
      ct9_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct9_rhs.shape[-1],
  )
  if ct9_lhs.shape != ct9_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct9_num_moduli = ct9_lhs.shape[-1]
  if hasattr(ct6, "moduli") and hasattr(ct7, "moduli"):
    if list(ct6.moduli)[:ct9_num_moduli] != list(ct7.moduli)[:ct9_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct9_moduli_src = getattr(ct6, "moduli", getattr(ct7, "moduli", v0.q_towers))
  if isinstance(ct9_moduli_src, (int, np.integer)):
    ct9_moduli_src = [ct9_moduli_src]
  ct9_moduli = jnp.array(
      list(ct9_moduli_src)[:ct9_num_moduli], dtype=jnp.uint64
  )
  ct9_sum = ct9_lhs.astype(jnp.uint64) + ct9_rhs.astype(jnp.uint64)
  ct9 = jnp.where(ct9_sum >= ct9_moduli, ct9_sum - ct9_moduli, ct9_sum).astype(
      jnp.uint32
  )
  ct10_lhs = ct9.polynomial if hasattr(ct9, "polynomial") else ct9
  ct10_rhs = ct8.polynomial if hasattr(ct8, "polynomial") else ct8
  ct10_lhs = ct10_lhs.reshape(
      ct10_lhs.shape[0],
      ct10_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct10_lhs.shape[-1],
  )
  ct10_rhs = ct10_rhs.reshape(
      ct10_rhs.shape[0],
      ct10_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct10_rhs.shape[-1],
  )
  if ct10_lhs.shape != ct10_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct10_num_moduli = ct10_lhs.shape[-1]
  if hasattr(ct9, "moduli") and hasattr(ct8, "moduli"):
    if list(ct9.moduli)[:ct10_num_moduli] != list(ct8.moduli)[:ct10_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct10_moduli_src = getattr(ct9, "moduli", getattr(ct8, "moduli", v0.q_towers))
  if isinstance(ct10_moduli_src, (int, np.integer)):
    ct10_moduli_src = [ct10_moduli_src]
  ct10_moduli = jnp.array(
      list(ct10_moduli_src)[:ct10_num_moduli], dtype=jnp.uint64
  )
  ct10_sum = ct10_lhs.astype(jnp.uint64) + ct10_rhs.astype(jnp.uint64)
  ct10 = jnp.where(
      ct10_sum >= ct10_moduli, ct10_sum - ct10_moduli, ct10_sum
  ).astype(jnp.uint32)
  ct11_arg = _ensure_poly(v0, ct10, v0.max_level)
  ct11 = v0.he_rot[v0.max_level, 3].rotate(ct11_arg)
  ct12_arg = _ensure_poly(v0, ct, v0.max_level)
  ct12_pt_ntt = (
      pt6.polynomial[0, 0, :, : ct12_arg.polynomial.shape[-1]]
      .reshape(ct12_arg.r, ct12_arg.c, ct12_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct12_ptct = v0.ptct_mul[v0.max_level]
  ct12_ptct.set_plaintext(ct12_pt_ntt)
  ct12 = ct12_ptct.mul(ct12_arg, use_bat=False)
  ct13_arg = _ensure_poly(v0, ct2, v0.max_level)
  ct13_pt_ntt = (
      pt7.polynomial[0, 0, :, : ct13_arg.polynomial.shape[-1]]
      .reshape(ct13_arg.r, ct13_arg.c, ct13_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct13_ptct = v0.ptct_mul[v0.max_level]
  ct13_ptct.set_plaintext(ct13_pt_ntt)
  ct13 = ct13_ptct.mul(ct13_arg, use_bat=False)
  ct14_lhs = ct12.polynomial if hasattr(ct12, "polynomial") else ct12
  ct14_rhs = ct13.polynomial if hasattr(ct13, "polynomial") else ct13
  ct14_lhs = ct14_lhs.reshape(
      ct14_lhs.shape[0],
      ct14_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct14_lhs.shape[-1],
  )
  ct14_rhs = ct14_rhs.reshape(
      ct14_rhs.shape[0],
      ct14_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct14_rhs.shape[-1],
  )
  if ct14_lhs.shape != ct14_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct14_num_moduli = ct14_lhs.shape[-1]
  if hasattr(ct12, "moduli") and hasattr(ct13, "moduli"):
    if (
        list(ct12.moduli)[:ct14_num_moduli]
        != list(ct13.moduli)[:ct14_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct14_moduli_src = getattr(
      ct12, "moduli", getattr(ct13, "moduli", v0.q_towers)
  )
  if isinstance(ct14_moduli_src, (int, np.integer)):
    ct14_moduli_src = [ct14_moduli_src]
  ct14_moduli = jnp.array(
      list(ct14_moduli_src)[:ct14_num_moduli], dtype=jnp.uint64
  )
  ct14_sum = ct14_lhs.astype(jnp.uint64) + ct14_rhs.astype(jnp.uint64)
  ct14 = jnp.where(
      ct14_sum >= ct14_moduli, ct14_sum - ct14_moduli, ct14_sum
  ).astype(jnp.uint32)
  ct15_arg = _ensure_poly(v0, ct14, v0.max_level)
  ct15 = v0.he_rot[v0.max_level, 6].rotate(ct15_arg)
  ct16_lhs = ct1.polynomial if hasattr(ct1, "polynomial") else ct1
  ct16_rhs = ct3.polynomial if hasattr(ct3, "polynomial") else ct3
  ct16_lhs = ct16_lhs.reshape(
      ct16_lhs.shape[0],
      ct16_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct16_lhs.shape[-1],
  )
  ct16_rhs = ct16_rhs.reshape(
      ct16_rhs.shape[0],
      ct16_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct16_rhs.shape[-1],
  )
  if ct16_lhs.shape != ct16_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct16_num_moduli = ct16_lhs.shape[-1]
  if hasattr(ct1, "moduli") and hasattr(ct3, "moduli"):
    if list(ct1.moduli)[:ct16_num_moduli] != list(ct3.moduli)[:ct16_num_moduli]:
      raise ValueError("ciphertext add modulus mismatch")
  ct16_moduli_src = getattr(ct1, "moduli", getattr(ct3, "moduli", v0.q_towers))
  if isinstance(ct16_moduli_src, (int, np.integer)):
    ct16_moduli_src = [ct16_moduli_src]
  ct16_moduli = jnp.array(
      list(ct16_moduli_src)[:ct16_num_moduli], dtype=jnp.uint64
  )
  ct16_sum = ct16_lhs.astype(jnp.uint64) + ct16_rhs.astype(jnp.uint64)
  ct16 = jnp.where(
      ct16_sum >= ct16_moduli, ct16_sum - ct16_moduli, ct16_sum
  ).astype(jnp.uint32)
  ct17_lhs = ct5.polynomial if hasattr(ct5, "polynomial") else ct5
  ct17_rhs = ct11.polynomial if hasattr(ct11, "polynomial") else ct11
  ct17_lhs = ct17_lhs.reshape(
      ct17_lhs.shape[0],
      ct17_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct17_lhs.shape[-1],
  )
  ct17_rhs = ct17_rhs.reshape(
      ct17_rhs.shape[0],
      ct17_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct17_rhs.shape[-1],
  )
  if ct17_lhs.shape != ct17_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct17_num_moduli = ct17_lhs.shape[-1]
  if hasattr(ct5, "moduli") and hasattr(ct11, "moduli"):
    if (
        list(ct5.moduli)[:ct17_num_moduli]
        != list(ct11.moduli)[:ct17_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct17_moduli_src = getattr(ct5, "moduli", getattr(ct11, "moduli", v0.q_towers))
  if isinstance(ct17_moduli_src, (int, np.integer)):
    ct17_moduli_src = [ct17_moduli_src]
  ct17_moduli = jnp.array(
      list(ct17_moduli_src)[:ct17_num_moduli], dtype=jnp.uint64
  )
  ct17_sum = ct17_lhs.astype(jnp.uint64) + ct17_rhs.astype(jnp.uint64)
  ct17 = jnp.where(
      ct17_sum >= ct17_moduli, ct17_sum - ct17_moduli, ct17_sum
  ).astype(jnp.uint32)
  ct18_lhs = ct17.polynomial if hasattr(ct17, "polynomial") else ct17
  ct18_rhs = ct15.polynomial if hasattr(ct15, "polynomial") else ct15
  ct18_lhs = ct18_lhs.reshape(
      ct18_lhs.shape[0],
      ct18_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct18_lhs.shape[-1],
  )
  ct18_rhs = ct18_rhs.reshape(
      ct18_rhs.shape[0],
      ct18_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct18_rhs.shape[-1],
  )
  if ct18_lhs.shape != ct18_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct18_num_moduli = ct18_lhs.shape[-1]
  if hasattr(ct17, "moduli") and hasattr(ct15, "moduli"):
    if (
        list(ct17.moduli)[:ct18_num_moduli]
        != list(ct15.moduli)[:ct18_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct18_moduli_src = getattr(
      ct17, "moduli", getattr(ct15, "moduli", v0.q_towers)
  )
  if isinstance(ct18_moduli_src, (int, np.integer)):
    ct18_moduli_src = [ct18_moduli_src]
  ct18_moduli = jnp.array(
      list(ct18_moduli_src)[:ct18_num_moduli], dtype=jnp.uint64
  )
  ct18_sum = ct18_lhs.astype(jnp.uint64) + ct18_rhs.astype(jnp.uint64)
  ct18 = jnp.where(
      ct18_sum >= ct18_moduli, ct18_sum - ct18_moduli, ct18_sum
  ).astype(jnp.uint32)
  ct19_lhs = ct16.polynomial if hasattr(ct16, "polynomial") else ct16
  ct19_rhs = ct18.polynomial if hasattr(ct18, "polynomial") else ct18
  ct19_lhs = ct19_lhs.reshape(
      ct19_lhs.shape[0],
      ct19_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct19_lhs.shape[-1],
  )
  ct19_rhs = ct19_rhs.reshape(
      ct19_rhs.shape[0],
      ct19_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct19_rhs.shape[-1],
  )
  if ct19_lhs.shape != ct19_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct19_num_moduli = ct19_lhs.shape[-1]
  if hasattr(ct16, "moduli") and hasattr(ct18, "moduli"):
    if (
        list(ct16.moduli)[:ct19_num_moduli]
        != list(ct18.moduli)[:ct19_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct19_moduli_src = getattr(
      ct16, "moduli", getattr(ct18, "moduli", v0.q_towers)
  )
  if isinstance(ct19_moduli_src, (int, np.integer)):
    ct19_moduli_src = [ct19_moduli_src]
  ct19_moduli = jnp.array(
      list(ct19_moduli_src)[:ct19_num_moduli], dtype=jnp.uint64
  )
  ct19_sum = ct19_lhs.astype(jnp.uint64) + ct19_rhs.astype(jnp.uint64)
  ct19 = jnp.where(
      ct19_sum >= ct19_moduli, ct19_sum - ct19_moduli, ct19_sum
  ).astype(jnp.uint32)
  ct20_arg = _ensure_poly(v0, ct19, v0.max_level)
  ct20 = v0.he_rescale[v0.max_level, v0.max_level - 1](ct20_arg)
  ct21_arg = _ensure_poly(v0, ct20, v0.max_level - 1)
  ct21_pt_ntt = (
      pt8.polynomial[0, 0, :, : ct21_arg.polynomial.shape[-1]]
      .reshape(ct21_arg.r, ct21_arg.c, ct21_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct21_ptct = v0.ptct_mul[v0.max_level - 1]
  ct21_ptct.set_plaintext(ct21_pt_ntt)
  ct21 = ct21_ptct.mul(ct21_arg, use_bat=False)
  ct22_arg = _ensure_poly(v0, ct19, v0.max_level)
  ct22 = v0.he_rot[v0.max_level, 1].rotate(ct22_arg)
  ct23_arg = _ensure_poly(v0, ct22, v0.max_level)
  ct23 = v0.he_rescale[v0.max_level, v0.max_level - 1](ct23_arg)
  ct24_arg = _ensure_poly(v0, ct23, v0.max_level - 1)
  ct24_pt_ntt = (
      pt9.polynomial[0, 0, :, : ct24_arg.polynomial.shape[-1]]
      .reshape(ct24_arg.r, ct24_arg.c, ct24_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct24_ptct = v0.ptct_mul[v0.max_level - 1]
  ct24_ptct.set_plaintext(ct24_pt_ntt)
  ct24 = ct24_ptct.mul(ct24_arg, use_bat=False)
  ct25_arg = _ensure_poly(v0, ct19, v0.max_level)
  ct25 = v0.he_rot[v0.max_level, 2].rotate(ct25_arg)
  ct26_arg = _ensure_poly(v0, ct25, v0.max_level)
  ct26 = v0.he_rescale[v0.max_level, v0.max_level - 1](ct26_arg)
  ct27_arg = _ensure_poly(v0, ct26, v0.max_level - 1)
  ct27_pt_ntt = (
      pt10.polynomial[0, 0, :, : ct27_arg.polynomial.shape[-1]]
      .reshape(ct27_arg.r, ct27_arg.c, ct27_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct27_ptct = v0.ptct_mul[v0.max_level - 1]
  ct27_ptct.set_plaintext(ct27_pt_ntt)
  ct27 = ct27_ptct.mul(ct27_arg, use_bat=False)
  ct28_arg = _ensure_poly(v0, ct20, v0.max_level - 1)
  ct28_pt_ntt = (
      pt11.polynomial[0, 0, :, : ct28_arg.polynomial.shape[-1]]
      .reshape(ct28_arg.r, ct28_arg.c, ct28_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct28_ptct = v0.ptct_mul[v0.max_level - 1]
  ct28_ptct.set_plaintext(ct28_pt_ntt)
  ct28 = ct28_ptct.mul(ct28_arg, use_bat=False)
  ct29_arg = _ensure_poly(v0, ct23, v0.max_level - 1)
  ct29_pt_ntt = (
      pt12.polynomial[0, 0, :, : ct29_arg.polynomial.shape[-1]]
      .reshape(ct29_arg.r, ct29_arg.c, ct29_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct29_ptct = v0.ptct_mul[v0.max_level - 1]
  ct29_ptct.set_plaintext(ct29_pt_ntt)
  ct29 = ct29_ptct.mul(ct29_arg, use_bat=False)
  ct30_arg = _ensure_poly(v0, ct26, v0.max_level - 1)
  ct30_pt_ntt = (
      pt13.polynomial[0, 0, :, : ct30_arg.polynomial.shape[-1]]
      .reshape(ct30_arg.r, ct30_arg.c, ct30_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct30_ptct = v0.ptct_mul[v0.max_level - 1]
  ct30_ptct.set_plaintext(ct30_pt_ntt)
  ct30 = ct30_ptct.mul(ct30_arg, use_bat=False)
  ct31_lhs = ct28.polynomial if hasattr(ct28, "polynomial") else ct28
  ct31_rhs = ct29.polynomial if hasattr(ct29, "polynomial") else ct29
  ct31_lhs = ct31_lhs.reshape(
      ct31_lhs.shape[0],
      ct31_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct31_lhs.shape[-1],
  )
  ct31_rhs = ct31_rhs.reshape(
      ct31_rhs.shape[0],
      ct31_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct31_rhs.shape[-1],
  )
  if ct31_lhs.shape != ct31_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct31_num_moduli = ct31_lhs.shape[-1]
  if hasattr(ct28, "moduli") and hasattr(ct29, "moduli"):
    if (
        list(ct28.moduli)[:ct31_num_moduli]
        != list(ct29.moduli)[:ct31_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct31_moduli_src = getattr(
      ct28, "moduli", getattr(ct29, "moduli", v0.q_towers)
  )
  if isinstance(ct31_moduli_src, (int, np.integer)):
    ct31_moduli_src = [ct31_moduli_src]
  ct31_moduli = jnp.array(
      list(ct31_moduli_src)[:ct31_num_moduli], dtype=jnp.uint64
  )
  ct31_sum = ct31_lhs.astype(jnp.uint64) + ct31_rhs.astype(jnp.uint64)
  ct31 = jnp.where(
      ct31_sum >= ct31_moduli, ct31_sum - ct31_moduli, ct31_sum
  ).astype(jnp.uint32)
  ct32_lhs = ct31.polynomial if hasattr(ct31, "polynomial") else ct31
  ct32_rhs = ct30.polynomial if hasattr(ct30, "polynomial") else ct30
  ct32_lhs = ct32_lhs.reshape(
      ct32_lhs.shape[0],
      ct32_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct32_lhs.shape[-1],
  )
  ct32_rhs = ct32_rhs.reshape(
      ct32_rhs.shape[0],
      ct32_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct32_rhs.shape[-1],
  )
  if ct32_lhs.shape != ct32_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct32_num_moduli = ct32_lhs.shape[-1]
  if hasattr(ct31, "moduli") and hasattr(ct30, "moduli"):
    if (
        list(ct31.moduli)[:ct32_num_moduli]
        != list(ct30.moduli)[:ct32_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct32_moduli_src = getattr(
      ct31, "moduli", getattr(ct30, "moduli", v0.q_towers)
  )
  if isinstance(ct32_moduli_src, (int, np.integer)):
    ct32_moduli_src = [ct32_moduli_src]
  ct32_moduli = jnp.array(
      list(ct32_moduli_src)[:ct32_num_moduli], dtype=jnp.uint64
  )
  ct32_sum = ct32_lhs.astype(jnp.uint64) + ct32_rhs.astype(jnp.uint64)
  ct32 = jnp.where(
      ct32_sum >= ct32_moduli, ct32_sum - ct32_moduli, ct32_sum
  ).astype(jnp.uint32)
  ct33_arg = _ensure_poly(v0, ct32, v0.max_level - 1)
  ct33 = v0.he_rot[v0.max_level - 1, 3].rotate(ct33_arg)
  ct34_arg = _ensure_poly(v0, ct20, v0.max_level - 1)
  ct34_pt_ntt = (
      pt14.polynomial[0, 0, :, : ct34_arg.polynomial.shape[-1]]
      .reshape(ct34_arg.r, ct34_arg.c, ct34_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct34_ptct = v0.ptct_mul[v0.max_level - 1]
  ct34_ptct.set_plaintext(ct34_pt_ntt)
  ct34 = ct34_ptct.mul(ct34_arg, use_bat=False)
  ct35_arg = _ensure_poly(v0, ct23, v0.max_level - 1)
  ct35_pt_ntt = (
      pt15.polynomial[0, 0, :, : ct35_arg.polynomial.shape[-1]]
      .reshape(ct35_arg.r, ct35_arg.c, ct35_arg.polynomial.shape[-1])
      .astype(jnp.uint32)
  )
  ct35_ptct = v0.ptct_mul[v0.max_level - 1]
  ct35_ptct.set_plaintext(ct35_pt_ntt)
  ct35 = ct35_ptct.mul(ct35_arg, use_bat=False)
  ct36_lhs = ct34.polynomial if hasattr(ct34, "polynomial") else ct34
  ct36_rhs = ct35.polynomial if hasattr(ct35, "polynomial") else ct35
  ct36_lhs = ct36_lhs.reshape(
      ct36_lhs.shape[0],
      ct36_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct36_lhs.shape[-1],
  )
  ct36_rhs = ct36_rhs.reshape(
      ct36_rhs.shape[0],
      ct36_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct36_rhs.shape[-1],
  )
  if ct36_lhs.shape != ct36_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct36_num_moduli = ct36_lhs.shape[-1]
  if hasattr(ct34, "moduli") and hasattr(ct35, "moduli"):
    if (
        list(ct34.moduli)[:ct36_num_moduli]
        != list(ct35.moduli)[:ct36_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct36_moduli_src = getattr(
      ct34, "moduli", getattr(ct35, "moduli", v0.q_towers)
  )
  if isinstance(ct36_moduli_src, (int, np.integer)):
    ct36_moduli_src = [ct36_moduli_src]
  ct36_moduli = jnp.array(
      list(ct36_moduli_src)[:ct36_num_moduli], dtype=jnp.uint64
  )
  ct36_sum = ct36_lhs.astype(jnp.uint64) + ct36_rhs.astype(jnp.uint64)
  ct36 = jnp.where(
      ct36_sum >= ct36_moduli, ct36_sum - ct36_moduli, ct36_sum
  ).astype(jnp.uint32)
  ct37_arg = _ensure_poly(v0, ct36, v0.max_level - 1)
  ct37 = v0.he_rot[v0.max_level - 1, 6].rotate(ct37_arg)
  ct38_lhs = ct21.polynomial if hasattr(ct21, "polynomial") else ct21
  ct38_rhs = ct24.polynomial if hasattr(ct24, "polynomial") else ct24
  ct38_lhs = ct38_lhs.reshape(
      ct38_lhs.shape[0],
      ct38_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct38_lhs.shape[-1],
  )
  ct38_rhs = ct38_rhs.reshape(
      ct38_rhs.shape[0],
      ct38_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct38_rhs.shape[-1],
  )
  if ct38_lhs.shape != ct38_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct38_num_moduli = ct38_lhs.shape[-1]
  if hasattr(ct21, "moduli") and hasattr(ct24, "moduli"):
    if (
        list(ct21.moduli)[:ct38_num_moduli]
        != list(ct24.moduli)[:ct38_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct38_moduli_src = getattr(
      ct21, "moduli", getattr(ct24, "moduli", v0.q_towers)
  )
  if isinstance(ct38_moduli_src, (int, np.integer)):
    ct38_moduli_src = [ct38_moduli_src]
  ct38_moduli = jnp.array(
      list(ct38_moduli_src)[:ct38_num_moduli], dtype=jnp.uint64
  )
  ct38_sum = ct38_lhs.astype(jnp.uint64) + ct38_rhs.astype(jnp.uint64)
  ct38 = jnp.where(
      ct38_sum >= ct38_moduli, ct38_sum - ct38_moduli, ct38_sum
  ).astype(jnp.uint32)
  ct39_lhs = ct27.polynomial if hasattr(ct27, "polynomial") else ct27
  ct39_rhs = ct33.polynomial if hasattr(ct33, "polynomial") else ct33
  ct39_lhs = ct39_lhs.reshape(
      ct39_lhs.shape[0],
      ct39_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct39_lhs.shape[-1],
  )
  ct39_rhs = ct39_rhs.reshape(
      ct39_rhs.shape[0],
      ct39_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct39_rhs.shape[-1],
  )
  if ct39_lhs.shape != ct39_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct39_num_moduli = ct39_lhs.shape[-1]
  if hasattr(ct27, "moduli") and hasattr(ct33, "moduli"):
    if (
        list(ct27.moduli)[:ct39_num_moduli]
        != list(ct33.moduli)[:ct39_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct39_moduli_src = getattr(
      ct27, "moduli", getattr(ct33, "moduli", v0.q_towers)
  )
  if isinstance(ct39_moduli_src, (int, np.integer)):
    ct39_moduli_src = [ct39_moduli_src]
  ct39_moduli = jnp.array(
      list(ct39_moduli_src)[:ct39_num_moduli], dtype=jnp.uint64
  )
  ct39_sum = ct39_lhs.astype(jnp.uint64) + ct39_rhs.astype(jnp.uint64)
  ct39 = jnp.where(
      ct39_sum >= ct39_moduli, ct39_sum - ct39_moduli, ct39_sum
  ).astype(jnp.uint32)
  ct40_lhs = ct39.polynomial if hasattr(ct39, "polynomial") else ct39
  ct40_rhs = ct37.polynomial if hasattr(ct37, "polynomial") else ct37
  ct40_lhs = ct40_lhs.reshape(
      ct40_lhs.shape[0],
      ct40_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct40_lhs.shape[-1],
  )
  ct40_rhs = ct40_rhs.reshape(
      ct40_rhs.shape[0],
      ct40_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct40_rhs.shape[-1],
  )
  if ct40_lhs.shape != ct40_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct40_num_moduli = ct40_lhs.shape[-1]
  if hasattr(ct39, "moduli") and hasattr(ct37, "moduli"):
    if (
        list(ct39.moduli)[:ct40_num_moduli]
        != list(ct37.moduli)[:ct40_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct40_moduli_src = getattr(
      ct39, "moduli", getattr(ct37, "moduli", v0.q_towers)
  )
  if isinstance(ct40_moduli_src, (int, np.integer)):
    ct40_moduli_src = [ct40_moduli_src]
  ct40_moduli = jnp.array(
      list(ct40_moduli_src)[:ct40_num_moduli], dtype=jnp.uint64
  )
  ct40_sum = ct40_lhs.astype(jnp.uint64) + ct40_rhs.astype(jnp.uint64)
  ct40 = jnp.where(
      ct40_sum >= ct40_moduli, ct40_sum - ct40_moduli, ct40_sum
  ).astype(jnp.uint32)
  ct41_lhs = ct38.polynomial if hasattr(ct38, "polynomial") else ct38
  ct41_rhs = ct40.polynomial if hasattr(ct40, "polynomial") else ct40
  ct41_lhs = ct41_lhs.reshape(
      ct41_lhs.shape[0],
      ct41_lhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct41_lhs.shape[-1],
  )
  ct41_rhs = ct41_rhs.reshape(
      ct41_rhs.shape[0],
      ct41_rhs.shape[1],
      v0._param_cache.r,
      v0._param_cache.c,
      ct41_rhs.shape[-1],
  )
  if ct41_lhs.shape != ct41_rhs.shape:
    raise ValueError("ciphertext add shape mismatch")
  ct41_num_moduli = ct41_lhs.shape[-1]
  if hasattr(ct38, "moduli") and hasattr(ct40, "moduli"):
    if (
        list(ct38.moduli)[:ct41_num_moduli]
        != list(ct40.moduli)[:ct41_num_moduli]
    ):
      raise ValueError("ciphertext add modulus mismatch")
  ct41_moduli_src = getattr(
      ct38, "moduli", getattr(ct40, "moduli", v0.q_towers)
  )
  if isinstance(ct41_moduli_src, (int, np.integer)):
    ct41_moduli_src = [ct41_moduli_src]
  ct41_moduli = jnp.array(
      list(ct41_moduli_src)[:ct41_num_moduli], dtype=jnp.uint64
  )
  ct41_sum = ct41_lhs.astype(jnp.uint64) + ct41_rhs.astype(jnp.uint64)
  ct41 = jnp.where(
      ct41_sum >= ct41_moduli, ct41_sum - ct41_moduli, ct41_sum
  ).astype(jnp.uint32)
  v20 = [None] * 1
  ct42_arg = _ensure_poly(v0, ct41, v0.max_level - 1)
  ct42 = v0.he_rescale[v0.max_level - 1, v0.max_level - 2](ct42_arg)
  v20[0] = ct42
  v21 = v20
  return v21


def matvec_chain(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
) -> np.ndarray:
  (v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14) = (
      matvec_chain__preprocessing(v0, v1)
  )
  v15 = matvec_chain__preprocessed(
      v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14
  )
  return v15


def matvec_chain__encrypt__arg0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = np.full(
      (
          1,
          8,
      ),
      0.000000e00,
      dtype=np.float32,
  )
  v6 = 0
  v7 = 1
  v8 = 8
  v9 = v5.copy()
  for v10 in range(0, 8):
    v12 = int(v10)
    v13 = v2[v12]
    v9[0, v12] = v13
  v15 = v9[0 : 0 + 1, 0 : 0 + 8].reshape(8)
  pt = v0.encode(v15)
  v0.public_key = v3
  ct_raw = v0.encrypt(pt)
  ct = _ensure_poly(v0, ct_raw)
  v16 = [ct]
  return v16


def matvec_chain__decrypt__result0(
    v0: ckks.CKKSContext,
    v1: dict,
    v2: np.ndarray,
    v3: np.ndarray,
) -> np.ndarray:
  v4 = 0
  v5 = 8
  v6 = 1
  v7 = 7
  v8 = 0
  v9 = np.full((8,), 0.000000e00, dtype=np.float32)
  ct = v2[0]
  v0.secret_key = v3
  pt_ct = _ensure_poly(v0, ct)
  _num_moduli = pt_ct.polynomial.shape[-1]
  _q_sub = list(getattr(pt_ct, "moduli", v0.q_towers))[:_num_moduli]
  _ct_for_dec = Polynomial(
      {
          "batch": pt_ct.polynomial.shape[0],
          "num_elements": pt_ct.polynomial.shape[1],
          "degree": v0.degree,
          "precision": 32,
          "num_moduli": _num_moduli,
          "degree_layout": (v0.degree,),
      },
      {"moduli": _q_sub},
  )
  _ct_for_dec.set_batch_polynomial(
      pt_ct.polynomial.reshape(
          pt_ct.polynomial.shape[0],
          pt_ct.polynomial.shape[1],
          v0.degree,
          _num_moduli,
      )
  )
  pt = v0.decrypt(_ct_for_dec)
  v10 = v0.decode(pt, is_ntt=False).real.reshape(1, 8)
  v11 = v9.copy()
  for v12 in range(0, 8):
    v14 = v7 - v12
    v15 = int(v14)
    v16 = v10[0, v15]
    v11[v15] = v16
  return v11


def matvec_identity__generate_crypto_context(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: dict,
) -> ckks.CKKSContext:
  params = {
      "degree": 16,
      "num_slots": 8,
      "batch": 1,
      "r": 4,
      "c": 4,
      "dnum": 3,
      "numEvalMult": 1,
      "scaling_factor": 35184372088832,
      "q_towers": [1073742881, 1073742721, 1073741441, 1073741857, 524353],
      "p_towers": [1073740609, 1073739937, 1073739649],
      "composite_degree": 1,
      "p": 30,
      "max_bits_in_word": 61,
      "max_bits_value": 9223372036854775295,
      "noise_scale_degree": 1,
      "CKKS_M_FACTOR": 1,
      "public_key": v0,
      "secret_key": v1,
      "evaluation_key": v2,
  }
  v3 = ckks.CKKSContext(params)
  return v3


def matvec_identity__configure_crypto_context(
    v0: ckks.CKKSContext,
):
  v0.program_initialization(
      total_hemul_levels=1,
      total_rotation_indices=[1, 2, 3, 6],
      dnum=3,
      r=4,
      c=4,
      batch=1,
  )
