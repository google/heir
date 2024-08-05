#ifndef LIB_DIALECT_RANDOM_IR_RANDOMOPS_TD_
#define LIB_DIALECT_RANDOM_IR_RANDOMOPS_TD_

include "lib/Dialect/Random/IR/RandomDialect.td"
include "lib/Dialect/Random/IR/RandomTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


class Random_Op<string mnemonic, list<Trait> traits = []> :
        Op<Random_Dialect, mnemonic, traits> {
  let cppNamespace = "::mlir::heir::random";
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Random_InitOp: Random_Op<"init_prng", [Pure]> {
  let summary = "Initializes the pseudorandom number generator with a seed.";
  let description = [{
    Initializes the PRNG with a seed. The seed is dynamically provided due to
    protocols that agree on shared randomness. The PRNG is used to initialized
    the random distributions such as the discrete gaussian distribution and the
    discrete uniform distribution. This initialization also takes as input a
    number of bits that are generated for each number value sampled (num_bits).
    For instance, a num_bits of 32 will mean that distributions will generate a
    32-bit integer value. We expect that the seed initialization is done statically
    and globally once per thread for all distributions; however, if multiple threads are
    generating randomness, then seed initialization should be done per thread;
    otherwise there is no guarantee of consistent behavior. Thread safety is so
    far not considered.
  }];

  let arguments = (ins
    SignlessIntegerLike:$seed,
    Builtin_IntegerAttr:$num_bits
  );
  let results = (outs PRNG:$output);
}

def Random_DiscreteGaussianDistributionOp : Random_Op<"discrete_gaussian_distribution", [Pure]> {
  let summary = "Initializes the Discrete Gaussian Distribution";
  let description = [{
    Initializes the Discrete Gaussian Distribution. The distribution is
    initialized with a mean and a standard deviation and pseudorandom generator
    that provides the source of the randomness.
  }];

  let arguments = (ins
    PRNG:$input,
    Builtin_IntegerAttr:$mean,
    ConfinedAttr<Builtin_IntegerAttr, [IntNonNegative]>:$stddev
  );
  let results = (outs Distribution:$output);
}

def Random_DiscreteUniformDistributionOp: Random_Op<"discrete_uniform_distribution", [Pure]> {
  let summary = "Initializes the Discrete Uniform Distribution";
  let description = [{
    Initializes the Discrete Uniform Distribution. The distribution is
    initialized with a minimum and a maximum value and pseudo random generator
    that provides the source of the randomness. The distribution is inclusive of
    the minimum and exclusive of the maximum.
  }];

  let arguments = (ins
    PRNG:$input,
    Builtin_IntegerAttr:$min,
    Builtin_IntegerAttr:$max
  );
  let results = (outs Distribution:$output);
  let assemblyFormat = "$input `{` `range` `=` `[` $min `,` $max `]` `}` attr-dict `:` `(` qualified(type($input)) `)` `->` type($output)";
  let hasVerifier = 1;
}

def Random_SampleOp : Random_Op<"sample"> {
  let summary = "Samples from a distribution";
  let description = [{
    Samples from the distribution to obtain a random value
    or tensor of values.
  }];
  let arguments = (ins Distribution:$input);
  let results = (outs SignlessIntegerLike:$output);
}

#endif  // LIB_DIALECT_RANDOM_IR_RANDOMOPS_TD_
