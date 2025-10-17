---
title: Ciphertext Management
weight: 9
---

On 2025-04-17, Hongren Zheng gave a talk overview of the ciphertext management
system in the HEIR working group meeting.
[The video can be found here](https://youtu.be/HHU6rCMxZRc?si=U_ePY5emqs6e4NoV&t=1631)
and [the slides can be found here](/slides/mgmt-2025-04-17.pdf)

## Introduction

To lower from user specified computation to FHE scheme operations, a compiler
must insert *ciphertext management* operations to satisfy various requirements
of the FHE scheme, like modulus switching, relinearization, and bootstrapping.
In HEIR, such operations are modeled in a scheme-agnostic way in the `mgmt`
dialect.

Taking the arithmetic pipeline as example: a program specified in high-level
MLIR dialects like `arith` and `linalg` is first transformed to an IR with only
`arith.addi/addf`, `arith.muli/mulf`, and `tensor_ext.rotate` operations. We
call this form the *secret arithmetic* IR.

Then management passes insert `mgmt` ops to support future lowerings to scheme
dialects like `bgv` and `ckks`. As different schemes have different management
requirement, they should be inserted in different styles.

We discuss each scheme below to show the design in HEIR. For RLWE schemes, we
all assume RNS instantiation.

## BGV

BGV is a leveled scheme where each level has a modulus $q_i$. The level is
numbered from $0$ to $L$ where $L$ is the input level and $0$ is the output
level. The core feature of BGV is that when the magnititude of the noise becomes
large (often caused by multiplication), a modulus switching operation from level
$i$ to level $i-1$ can be inserted to reduce the noise to a "constant" level. In
this way, BGV can support a circuit of multiplicative depth $L$.

### BGV: Relinearization

HEIR initially inserts relinearization ops immediately after each multiplication
to keep ciphertext dimension "linear". A later relinearization optimization pass
relaxes this requirement, and uses an integer linear program to decide when to
relinearize. See [Optimizing Relinearization](/docs/design/relinearization_ilp/)
for more details.

### BGV: Modulus switching

There are several techniques to insert modulus switching ops.

For the example circuit `input -> mult -> mult -> output`, the insertion result
could be one of

1. After multiplication: `input -> (mult -> ms) -> (mult -> ms) -> output`

1. Before multiplication: `input -> (mult) -> (ms -> mult) -> (ms -> output)`

1. Before multiplication (including the first multiplication):
   `input -> (ms -> mult) -> (ms -> mult) -> (ms -> output)`

The first strategy is from the BGV paper, the second and third strategies are
from OpenFHE, which correspond to the `FLEXIBLEAUTO` mode and `FLEXIBLEAUTOEXT`
mode, respectively.

The first strategy is conceptually simpler, yet other policies have the
advantage of smaller noise growth. In latter policies, by delaying the modulus
switch until just before multiplication, the noise from other operations between
multiplications (like rotation/relinearization) also benefit from the noise
reduction of a modulus switch.

Note that, as multiplication has two operands, the actual circuit for the latter
two policies is `mult(ms(ct0), ms(ct1))`, whereas in the first policy the
circuit is `ms(mult(ct0, ct1))`.

The third policy has one more switching op than the others, so it will need one
more modulus.

There are also other insertion strategy like inserting it dynamically based on
current noise (see HElib) or lazy modulus switching. Those are not implemented.

### BGV: Scale management

For the original BGV scheme, it is required to have $qi \\equiv 1 \\pmod{t}$
where $t$ is the plaintext modulus. However in practice such requirement will
make the choice of $q_i$ too constrained. In the GHS variant, this condition is
removed, with the price of scale management needed.

Modulus switching from level $i$ to level $i-1$ is essentially dividing (with
rounding) the ciphertext by $q_i$, hence dividing the noise and payload message
inside by $q_i$. The message $m$ can often be written as $\[m\]\_t$, the coset
representative of `m` $\\mathbb{Z}/t\\mathbb{Z}$. Then by dividing of $q_i$
produces a result message $\[m \\cdot q_i^{-1}\]\_t$.

Note that when $qi \\equiv 1 \\pmod{t}$, the result message is the same as the
original message. However, in the GHS variant this does not always hold, so we
call the introduced factor of $\[q^{-1}\]\_t$ the *scale* of the message. HEIR
needs to record and manage it during compilation. When decrypting the scale must
be removed to obtain the right message.

Note that, for messages $m_0$ and $m_1$ of different scale $a$ and $b$, we
cannot add them directly because $\[a \\cdot m_0 + b \\cdot m_1\]\_t$ does not
always equal $\[m_0 + m_1\]\_t$. Instead we need to adjust the scale of one
message to match the other, so $\[b \\cdot m_0 + b \\cdot m_1\]\_t = \[b \\cdot
(m_0 + m_1)\]\_t$. Such adjustment could be done by multiplying $m_0$ with a
constant $\[b \\cdot a^{-1}\]\_t$. This adjustment is not for free, and
increases the ciphertext noise.

As one may expect, different modulus switching insertion strategies affect
message scale differently. For $m_0$ with scale $a$ and $m_1$ with scale $b$,
the result scale would be

1. After multiplication: $\[ab / qi\]\_t$.

1. Before multiplication: $\[a / qi \\cdot b / qi\]\_t = \[ab / (qi^2)\]\_t$.

This is messy enough. To ease the burden, we can impose additional requirement:
mandate a constant scale $\\Delta_i$ for all ciphertext at level $i$. This is
called the *level-specific scaling factor*. With this in mind, addition within
one level can happen without caring about the scale.

1. After multiplication: $\\Delta\_{i-1} = \[\\Delta_i^2 / qi\]\_t$

1. Before multiplication: $\\Delta\_{i-1} = \[\\Delta_i^2 / (qi^2)\]\_t$

### BGV: Cross Level Operation

With the level-specific scaling factor, one may wonder how to perform addition
and multiplication of ciphertexts on different levels. This can be done by
adjusting the level and scale of the ciphertext at the higher level.

The level can be easily adjusted by dropping the extra limbs, and scale can be
adjusted by multiplying a constant, but because multiplying a constant will
incur additional noise, the procedure becomes the following:

1. Assume the level and scale of two ciphertexts are $l_1$ and $l_2$, $s_1$ and
   $s_2$ respectively. WLOG assume $l_1 > l_2$.

1. Drop $l_1 - l_2 - 1$ limbs for the first ciphertext to make it at level $l_2
   \+ 1$, if those extra limbs exist.

1. Adjust scale from $s_1$ to $s_2 \\cdot q\_{l_2 + 1}$ by multiplying $\[s_2
   \\cdot q\_{l_2 + 1} / s1\]\_t$ for the first ciphertext.

1. Modulus switch from $l_2 + 1$ to $l_2$, producing scale $s_2$ for the first
   ciphertext and its noise is controlled.

### BGV: Implementation in HEIR

In HEIR the different modulus switching policy is controlled by the pass option
for `--secret-insert-mgmt-bgv`. The pass defaults to the "Before Multiplication"
policy. If user wants other policy, the `after-mul` or
`before-mul-include-first-mul` option may be used. The `mlir-to-bgv` pipeline
option `modulus-switch-before-first-mul` corresponds to the latter option.

The `secret-insert-mgmt` pass is also responsible for managing cross-level
operations. However, as the scheme parameters are not generated at this point,
the concrete scale could not be instantiated so some placeholder operations are
inserted.

After the modulus switching policy is applied, the `generate-param-bgv` pass
generates scheme parameters. Optionally, user could skip this pass by manually
providing scheme parameter as an attribute at module level.

Then `populate-scale-bgv` comes into play by using the scheme parameters to
instantiate concrete scale, and turn those placeholder operations into concrete
multiplication operation.

## CKKS

CKKS is a leveled scheme where each level has a modulus $q_i$. The level is
numbered from $0$ to $L$ where $L$ is the input level and $0$ is the output
level. CKKS ciphertext contains a scaled message $\\Delta m$ where $\\Delta$
takes some value like $2^40$ or $2^80$. After multiplication of two messages,
the scaling factor $\\Delta'$ will become larger, hence some kind of management
policy is needed in case it blows up. Contrary to BGV where modulus switching is
used for noise management, in CKKS modulus switching from level $i$ to level
$i-1$ can divide the scaling factor $\\Delta$ by the modulus $q_i$.

The management of CKKS is similar to BGV above in the sense that their strategy
are the similar and uses similar code base. However, BGV scale management is
internal and users are not required to concern about it, while CKKS scale
management is visible to user as it affects the precision. One notable
difference is that, for "Before multiplication (including the first
multiplication)" modulus switching policy, the user input should be encoded at
$\\Delta^2$ or higher, as otherwise the first modulus switching (or rescaling in
CKKS term) will rescale $\\Delta$ to $1$, rendering full precision loss.

<!-- mdformat global-off -->
