#ifndef INCLUDE_INTERFACES_NOISEINTERFACES_TD_
#define INCLUDE_INTERFACES_NOISEINTERFACES_TD_

include "mlir/IR/OpBase.td"

def NoisePropagationInterface : OpInterface<"NoisePropagationInterface"> {
  let description = [{
    Declares that an operation produces results with noise, and provides an
    interface for passes to compute bounds on the noise in the results
    from the input noises.

    Here "noise" is defined as the (perhaps upper-bounded) variance of a
    Gaussian distribution centered at zero.
  }];
  let cppNamespace = "::mlir::heir";

  let methods = [
    InterfaceMethod<[{
      Infer the noise distribution of the result of this op given the distributions
      of its inputs.

      All noise distributions are assumed to be Gaussian centered at zero, and
      so the inputs and results are represented by their variances.

      For each result value or block argument (that isn't a branch argument,
      since the dataflow analysis handles those case), the method should call
      `setValueNoise` with that `Value` as an argument. When `setValueNoise`
      is not called for some value, the analysis will raise an error.

      `argNoises` contains one `int64_t` for each operand to the op in ODS
       order. Operands that don't have a prior noise associated with them
       will have this value set to zero.
    }],
    "void", "inferResultNoise", (ins
      "::llvm::ArrayRef<int64_t>":$argNoises,
      "::mlir::heir::SetNoiseFn":$setValueNoise)
  >];
}


#endif  // INCLUDE_INTERFACES_NOISEINTERFACES_TD_
