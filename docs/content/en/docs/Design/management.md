---
title: Ciphertext Management
weight: 9
---

To lower from user specified computation to FHE scheme operations, compiler
needs to insert proper _management_ operations to satisfy various requirements
imposed by FHE scheme, like modulus switching, relinearization, bootstrapping.
In HEIR, such operations are modeled in a scheme-agnostic way in `mgmt` dialect.

Taking the arithmetic pipeline as example: user input specified in various MLIR
dialect like `arith` and `linalg` are first transformed by optimizations like
SIMD vectorizer passes and packing passes, to an IR with only
addition/multiplication/rotation operations. We call it _secret arithmetic_ IR.

Then management passes need to insert operations from `mgmt` dialect to _secret
arithmetic_ IR for future lowering to scheme dialect like `bgv` and `ckks`. As
different schemes have different management requirement, they should be inserted
in different styles.

We discuss each scheme below to show the design in HEIR. For RLWE schemes, we
all assume RNS instantiation.

## BGV

BGV is a leveled scheme where each level has a modulus `qi`. The level is
numbered from `0` to `L` where `L` is the input level and `0` is the output
level. The core feature of BGV is that when the magnititude of the noise becomes
large (often caused by multiplication) , a modulus switching operation from
level `i` to level `i-1` can be inserted to reduce the noise to a "constant"
level. In this way, BGV can support a circuit of multiplicative depth `L`
instead of just `L` multiplications.

### BGV: Relinearization

HEIR insert relinearization immediately after each multiplication to keep
ciphertext "linear", and relies on lazy relinearization pass to optimize it.

Detail on why relinearization is needed could be found at the "Optimizing
Relinearization" design document.

### BGV: Modulus switching

As introduced above, modulus switching should be done at proper time. There are
several styles of inserting modulus switching.

For the example circuit `input -> mult -> mult -> output`, the insertion result
would be

1. After multiplication: `input -> (mult -> ms) -> (mult -> ms) -> output`

1. Before multiplication: `input -> (mult) -> (ms -> mult) -> (ms -> output)`

1. Before multiplication (including the first multiplication):
   `input -> (ms -> mult) -> (ms -> mult) -> (ms -> output)`

The first strategy is from the BGV paper, the second and third strategies are
from OpenFHE, which correspond to the `FLEXIBLEAUTO` mode and `FLEXIBLEAUTOEXT`
mode, respectively.

The first strategy is conceptually simpler, yet other policies have the
advantage of smaller noise growth. In latter policies, at the beginning of each
level, the noise level would be the level of multiplication instead of modulus
switching, so the error incured by other operations (relinearization/rotation)
in the same level could be hidden.

Note that, as multiplication has two operands, the actual circuit for the latter
two policies is `mult(ms(ct0), ms(ct1))`, whereas in the first policy the
circuit is `ms(mult(ct0, ct1))`.

The third policy has one more modulus switching than others, so it will need one
more modulus.

There are also other insertion strategy like inserting it dynamically based on
current noise (see HElib) or lazy rescaling. Those are not implemented.

### BGV: Scale management

For the original BGV scheme, it is required to have `qi = 1 (mod t)` where `t`
is the plaintext modulus. However in practice such requirement will make the
choice of `qi` much scarse. So in the GHS variant, this condition is removed,
with the price of scale management needed.

Modulus switching from level `i` to level `i-1` is essentially dividing (with
rounding) the ciphertext by `qi`, hence dividing the noise and payload message
inside by `qi`. The message `m` can often be written as `[m]_t` as the
computation happens in `Zt`, then by dividing of `qi` the result message would
be `[m / qi]_t`.

Note that when `qi = 1 (mod t)`, the result message is the same as the original
message. However, in GHS variant such condition does not always hold, so we
would call the `[q^{-1}]_t` part as the _scale_ of the message and compiler
needs to record and manage it during computation. And when decrypting such scale
must be removed to obtain the original message.

Note that, for messages `m0` and `m1` of different scale `a` and `b`, we could
not add them directly because `[a * m0 + b * m1]_t` is often not the expected
result for user. Instead we need to adjust the scale of one message to match
another so `[b * m0 + b * m1]_t = [b * (m0 + m1)]_t`. Such adjustment could be
done by multiplying `m0` with a constant `[b / a]_t`. This adjustment is not for
free as it will also amplify the noise.

As one may expect, different modulus switching insertion strategy would affect
the scale on the message. For `m0` with scale `a` and `input1` with scale `b`,
the result scale would be

1. After multiplication: `[ab / qi]_t`.

1. Before multiplication: `[a / qi * b / qi]_t` = `[ab / (qi^2)]_t`.

This is messy enough. To ease the burden, we can impose additional requirement:
Mandate a constant scale `Delta_i` for each level. This is called
_level-specific scaling factor_. With this in mind, addition within one level
can happen without caring about the scale, and the scale of the next level is
also the same.

1. After multilication: `Delta_{i-1} = [Delta_i^2 / qi]_t`

1. Before multilication: `Delta_{i-1} = [Delta_i^2 / (qi^2)]_t`

### BGV: Cross Level Operation

With the level-specific scaling factor, one may wonder how could addition and
multiplication of ciphertext of different levels happen as they have different
scaling factor. This can be solved by adjusting the level and scale of the
ciphertext of higher level.

The level can be easily adjusted by dropping the extra limbs, and scale can be
adjusted by multiplying a constant, but because multiplying a constant will
incur additional noise, the procedure becomes the following:

1. Assume the level and scale of two ciphertexts are `l1` and `l2`, `s1` and
   `s2` respectively. WLOG assume `l1 > l2`.

1. Drop `l1 - l2 - 1` limbs for the first ciphertext to make it at level
   `l2 + 1`, if those extra limbs exist.

1. Adjust scale from `s1` to `s2 * q_{l2 + 1}` by multiplying
   `[s2 * q_{l2 + 1} / s1]_t` for the first ciphertext.

1. Modulusing switching from `l2 + 1` to `l2`, the scale becomes `s2` for the
   first ciphertext.
