---
title: Optimizing relinearization
weight: 10
---

This document outlines the integer linear program model used in the
[`optimize-relinearization`](https://heir.dev/docs/passes/optimizerelinearizationpasses/#-optimize-relinearization)
pass.

## Background

In vector/arithmetic FHE, RLWE ciphertexts often have the form $\\mathbf{c} =
(c_0, c_1)$, where the details of how $c_0$ and $c_1$ are computed depend on the
specific scheme. However, in most of these schemes, the process of decryption
can be thought of as taking a dot product between the vector $\\mathbf{c}$ and a
vector $(1, s)$ containing the secret key $s$ (followed by rounding).

In such schemes, the homomorphic multiplication of two ciphertexts $\\mathbf{c}
= (c_0, c_1)$ and $\\mathbf{d} = (d_0, d_1)$ produces a ciphertext $\\mathbf{f}
= (f_0, f_1, f_2)$. This triple can be decrypted by taking a dot product with
$(1, s, s^2)$.

With this in mind, each RLWE ciphertext $\\mathbf{c}$ has an associated _key
basis_, which is the vector $\\mathbf{s_c}$ whose dot product with $\\mathbf{c}$
decrypts it.

Usually a larger key basis is undesirable. For one, operations in a higher key
basis are more expensive and have higher rates of noise growth. Repeated
multiplications exponentially increase the length of the key basis. So to avoid
this, an operation called _relinearization_ was designed that converts a
ciphertext from a given key basis back to $(1, s)$. Doing this requires a set of
_relinearization keys_ to be provided by the client and stored by the server.

In general, key bases can be arbitrary. Rotation of an RLWE ciphertext by a
shift of $k$, for example, first applies the automorphism $x \\mapsto x^k$. This
converts the key basis from $(1, s)$ to $(1, s^k)$, and more generally maps $(1,
s, s^2, \\dots, s^d) \\mapsto (1, s^k, s^{2k}, \\dots, s^{kd})$. Most FHE
implementations post-compose this automorphism with a key switching operation to
return to the linear basis $(1, s)$. Similarly, multiplication can be defined
for two key bases $(1, s^n)$ and $(1, s^m)$ (with $n \< m$) to produce a key
basis $(1, s^n, s^m, s^{n+m})$. By a combination of multiplications and
rotations (without ever relinearizing or key switching), ciphertexts with a
variety of strange key bases can be produced.

Most FHE implementations do not permit wild key bases because each key switch
and relinearization operation (for each choice of key basis) requires additional
secret key material to be stored by the server. Instead, they often enforce that
rotation has key-switching built in, and multiplication relinearizes by default.

That said, many FHE implementations do allow for the relinearization operation
to be deferred. A useful such situation is when a series of independent
multiplications are performed, and the results are added together. Addition can
operate in any key basis (though depending on the backend FHE implementation's
details, all inputs may require the same key basis, cf.
[Optional operand agreement](#optional-operand-agreement)), and so the
relinearization op that follows each multiplication can be deferred until after
the additions are complete, at which point there is only one relinearization to
perform. This technique is usually called _lazy relinearization_. It has the
benefit of avoiding expensive relinearization operations, as well as reducing
noise growth, as relinearization adds noise to the ciphertext, which can further
reduce the need for bootstrapping.

In much of the literature, lazy relinearization is applied manually. See for
example
[Blatt-Gusev-Polyakov-Rohloff-Vaikuntanathan 2019](https://eprint.iacr.org/2019/223)
and [Lee-Lee-Kim-Kim-No-Kang 2020](https://eprint.iacr.org/2020/1549). In some
compiler projects, such as the [EVA compiler](https://eprint.iacr.org/2021/1505)
relinearization is applied automatically via a heuristic, either "eagerly"
(immediately after each multiplication op) or "lazily," deferred as late as
possible.

## The `optimize-relinearization` pass

In HEIR, relinearization placement is implemented via a mixed-integer linear
program (ILP). It is intended to be more general than a lazy relinearization
heuristic, and certain parameter settings of the ILP reproduce lazy
relinearization.

The `optimize-relinearization` pass starts by deleting all relinearization
operations from the IR, solves the ILP, and then inserts relinearization ops
according to the solution. This implies that the input IR to the ILP has no
relinearization ops in it already.

## Model specification

The ILP model fits into a family of models that is sometimes called
"state-dynamics" models, in that it has "state" variables that track a quantity
that flows through a system, as well as "decision" variables that control
decisions to change the state at particular points. A brief overview of state
dynamics models can be found
[here](https://buttondown.com/j2kun/archive/modeling-state-in-linear-programs/)

In this ILP, the "state" value is the degree of the key basis. I.e., rather than
track the entire key basis, we assume the key basis always has the form $(1, s,
s^2, \\dots, s^k)$ and track the value $k$. The index tracking state is SSA
value, and the decision variables are whether to relinearize.

### Variables

Define the following variables:

- For each operation $o$, $R_o \\in { 0, 1 }$ defines the decision to
  relinearize the result of operation $o$. Relinearization is applied if and
  only if $R_o = 1$.
- For each SSA value $v$, $\\textup{KB}\_v$ is a continuous variable
  representing the degree of the key basis of $v$. For example, if the key basis
  of a ciphertext is $(1, s)$, then $\\textup{KB}\_v = 1$. If $v$ is the result
  of an operation $o$, $\\textup{KB}\_v$ is the key basis of the result of $o$
  _after_ relinearization has been optionally applied to it, depending on the
  value of the decision variable $R_o$.
- For each SSA value $v$ that is an operation result, $\\textup{KB}^{br}\_v$ is
  a continuous variable whose value represents the key basis degree of $v$
  _before_ relinearization is applied (`br` = "before relin"). These SSA values
  are mainly for _after_ the model is solved and relinearization operations need
  to be inserted into the IR. Here, type conflicts require us to reconstruct the
  key basis degree, and saving the values allows us to avoid recomputing the
  values.

Each of the key-basis variables is bounded from above by a parameter
`MAX_KEY_BASIS_DEGREE` that can be used to impose hard limits on the key basis
size, which may be required if generating code for a backend that does not
support operations over generalized key bases.

### Objective

The objective is to minimize the number of relinearization operations, i.e.,
$\\min \\sum_o R_o$.

TODO(#1018): update docs when objective is generalized.

### Constraints

#### Simple constraints

The simple constraints are as follows:

- Initial key basis degree: For each block argument, $\\textup{KB}\_v$ is fixed
  to equal the `dimension` parameter on the RLWE ciphertext type.
- Special linearized ops: `bgv.rotate` and `func.return` require linearized
  inputs, i.e., $\\textup{KB}\_{v_i} = 1$ for all inputs $v_i$ to these
  operations.
- Before relinearization key basis: for each operation $o$ with operands $v_1,
  \\dots, v_k$, constrain $\\textup{KB}^{br}\_{\\textup{result}(o)} =
  f(\\textup{KB}\_{v_1}, \\dots, \\textup{KB}\_{v_k})$, where $f$ is a
  statically known linear function. For multiplication $f$ it addition, and for
  all other ops it is the projection onto any input, since multiplication is the
  only op that increases the degree, and all operands are constrained to have
  equal degree.

#### Optional operand agreement

There are two versions of the model, one where the an operation requires the
input key basis degrees of each operand to be equal, and one where differing key
basis degrees are allowed.

This is an option because the model was originally implemented under the
incorrect assumption that CPU backends like OpenFHE and Lattigo require the key
basis degree operands to be equal for ops like ciphertext addition. When we
discovered this was not the case, we generalized the model to support both
cases, in case other backends do have this requirement.

When operands must have the same key basis degree, then for each operation with
operand SSA values $v_1, \\dots, v_k$, we add the constraint
$\\textup{KB}\_{v_1} = \\dots = \\textup{KB}\_{v_k}$, i.e., all key basis inputs
must match.

When operands may have different key basis degrees, we instead add the
constraint that each operation result key basis degree (before relinearization)
is at least as large as the max of all operand key basis degrees. For all $i$,
$\\textup{KB}\_{\\textup{result}(o)}^{br} \\geq \\textup{KB}\_{v_i}$. Note that
we are relying on an implicit behavior of the model to ensure that, even if the
solver chooses key basis degree variables for these op results larger than the
max of the operand degrees, the resulting optimal solution is the same.

TODO(#1018): this will change to a more principled approach when the objective
is generalized

#### Impact of relinearization choices on key basis degree

The remaining constraints control the dynamics of how the key basis degree
changes as relinearizations are inserted.

They can be thought of as implementing this (non-linear) constraint for each
operation $o$:

\\\[ \\textup{KB}\_{\\textup{result}(o)} = \\begin{cases}
\\textup{KB}^{br}\_{\\textup{result(o)}} & \\text{ if } R_o = 0 \\\\ 1 & \\text{
if } R_o = 1 \\end{cases} \\\]

Note that $\\textup{KB}^{br}\_{\\textup{result}(o)}$ is constrained by one of
the simple constraints to be a linear expression containing key basis variables
for the operands of $o$. The conditional above cannot be implemented directly in
an ILP. Instead, one can implement it via four constraints that effectively
linearize (in the sense of making non-linear constraints linear) the multiplexer
formula

\\\[ \\textup{KB}\_{\\textup{result}(o)} = (1 - R_o) \\cdot
\\textup{KB}^{br}\_{\\textup{result}(o)} + R_o \\cdot 1 \\\]

(Note the above is not linear because in includes the product of two variables.)
The four constraints are:

\\\[ \\begin{aligned} \\textup{KB}\_\\textup{result}(o) &\\geq \\textup{ R}\_o
\\\
\\textup{KB}\_\\textup{result}(o) &\\leq 1 + C(1 – \\textup{R}\_o)
\\\
\\textup{KB}\_\\textup{result}(o) &\\geq
\\textup{KB}^{br}\_{\\textup{result}(o)} – C \\textup{ R}\_o
\\\
\\textup{KB}\_\\textup{result}(o) &\\leq
\\textup{KB}^{br}\_{\\textup{result}(o)} + C \\textup{ R}\_o \\\
\\end{aligned}
\\\]

Here $C$ is a constant that can be set to any value larger than
`MAX_KEY_BASIS_DEGREE`. We set it to 100.

Setting $R_o = 0$ makes constraints 1 and 2 trivially satisfied, while
constraints 3 and 4 enforce the equality $\\textup{KB}\_{\\textup{result}(o)} =
\\textup{KB}^{br}\_{\\textup{result}(o)}$. Likewise, setting $R_o = 1$ makes
constraints 3 and 4 trivially satisfied, while constraints 1 and 2 enforce the
equality $\\textup{KB}\_{\\textup{result}(o)} = 1$.

## Notes

- ILP performance scales roughly with the number of integer variables. The
  formulation above only requires the decision variable to be integer, and the
  initialization and constraints effectively force the key basis variables to be
  integer. As a result, the solve time of the above ILP should scale with the
  number of ciphertext-handling ops in the program.

<!-- mdformat global-off -->
