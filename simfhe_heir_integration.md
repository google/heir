# HEIR Integration with SimFHE

## Integration Overview

HEIR takes high-level programs and compiles them to programs over FHE
operations, with the ability to target them to a variety of backends (e.g.
OpenFHE, Lattigo, TFHE-rs, etc).

### Fork Changes Summary

Alexander Viand's fork (https://github.com/alexanderviand/SimFHE) makes minimal
changes:

1. **profiler.py**: Added "generated" to `DECORATION_LIST` (line 21)

   ```python
   DECORATION_LIST = [
       ...
       "poly",
       "generated",  # Added this line
   ]
   ```

1. **requirements.txt**: Added explicit dependencies

   ```
   tqdm
   tabulate
   ```

1. **.gitignore**: Added Python-specific ignores

   ```
   venv
   __pycache__
   ```

1. **Code formatting**: Minor fix to assert statement (removed parentheses)

These changes enable SimFHE to recognize and execute HEIR-generated code
modules.

### Current Integration Status

- **Supported**: CKKS operations (add, sub, mul, rotate, relinearize)
- **Backend**: SimFHE Python API via code generation
- **Status**: Experimental - functional but with uncertainties about correct
  usage

## Example: CKKS Computation

### Original MLIR (High-level)

```mlir
// Computes: result = (x * x) + (2 * x * y) - y
func.func @polynomial_evaluation(%x: tensor<16xf32> {secret.secret},
                                %y: tensor<16xf32> {secret.secret})
                                -> tensor<16xf32> {
  %x_squared = arith.mulf %x, %x : tensor<16xf32>
  %xy = arith.mulf %x, %y : tensor<16xf32>
  %c2 = arith.constant dense<2.0> : tensor<16xf32>
  %two_xy = arith.mulf %xy, %c2 : tensor<16xf32>
  %sum1 = arith.addf %x_squared, %two_xy : tensor<16xf32>
  %sum2 = arith.addf %sum1, %y : tensor<16xf32>
  %c3 = arith.constant dense<3.0> : tensor<16xf32>
  %result = arith.subf %sum2, %c3 : tensor<16xf32>
  return %result : tensor<16xf32>
}
```

### Generated SimFHE Python

```python
import params
import evaluator
from perf_counter import PerfCounter
from experiment import run_mutiple, print_table, Target

def polynomial_eval(ct, ct1, scheme_params : params.SchemeParams):
  stats = PerfCounter()
  stats += evaluator.multiply(ct, scheme_params.arch_param)
  ct2 = ct
  stats += evaluator.key_switch(ct2, scheme_params.fresh_ctxt, scheme_params.arch_param)
  ct3 = ct2
  stats += evaluator.multiply(ct, scheme_params.arch_param)
  ct4 = ct
  stats += evaluator.key_switch(ct4, scheme_params.fresh_ctxt, scheme_params.arch_param)
  ct5 = ct4
  stats += evaluator.add(ct5, scheme_params.arch_param)
  ct6 = ct5
  stats += evaluator.add(ct3, scheme_params.arch_param)
  ct7 = ct3
  stats += evaluator.add(ct7, scheme_params.arch_param)
  ct8 = ct7
  stats += evaluator.rotate(ct8, scheme_params.arch_param)
  ct9 = ct8
  stats += evaluator.add(ct8, scheme_params.arch_param)
  ct10 = ct8
  return stats
```

### Operation Mapping Table

| CKKS Operation     | SimFHE API Call              | Notes                    |
| ------------------ | ---------------------------- | ------------------------ |
| `ckks.add`         | `evaluator.add()`            | Direct mapping           |
| `ckks.sub`         | `evaluator.add()`            | SimFHE lacks subtraction |
| `ckks.mul`         | `evaluator.multiply()`       | Direct mapping           |
| `ckks.mul_plain`   | `evaluator.multiply_plain()` | Direct mapping           |
| `ckks.negate`      | `evaluator.multiply_plain()` | Multiply by -1           |
| `ckks.rotate`      | `evaluator.rotate()`         | Direct mapping           |
| `ckks.relinearize` | `evaluator.key_switch()`     | Uses fresh_ctxt as key   |

## Technical Details

### API Usage Patterns

1. **PerfCounter accumulation**: Each operation adds to a cumulative
   `PerfCounter`
1. **Parameter passing**: All operations receive `scheme_params.arch_param`
1. **Key switching**: Relinearization uses `scheme_params.fresh_ctxt` as the key
   parameter
1. **Variable naming**: Generated variables follow pattern `ct`, `ct1`, `ct2`,
   etc.

### Parameter Handling

Currently, the emitter uses the same scheme parameters for all workloads:

- `params.Alg_benchmark_baseline`
- `params.Alg_benchmark_mod_down_merge`
- `params.Alg_benchmark_mod_down_hoist`
- `params.BEST_PARAMS`

### Workarounds & Assumptions

1. **Subtraction**: Mapped to addition (SimFHE limitation)
1. **Negation**: Implemented as multiplication by -1
1. **Relinearization key**: Assumes key similar to `fresh_ctxt`
1. **Bootstrap/Rescale**: Not yet implemented in emitter
1. **Variable reuse**: Generated code reassigns variables (e.g., `ct2 = ct`)

### Areas of Uncertainty

1. **Key selection for relinearization**: Warning indicates uncertainty about
   correct key usage
1. **Ciphertext dimension tracking**: Currently not tracking dimension changes
1. **Modulus chain management**: Not explicitly handled
1. **Memory optimization**: Variable reassignment pattern may not reflect actual
   memory usage

## Questions & Feedback

### API Clarifications Needed

1. **Relinearization keys**: What is the correct way to specify relinearization
   keys in SimFHE?
1. **Subtraction operation**: Is there a plan to add native subtraction support?
1. **Bootstrap operation**: How should bootstrapping be modeled in SimFHE?
1. **Ciphertext metadata**: Should we track/pass ciphertext dimensions and
   levels?

### Feature Requests

1. **Native subtraction operation**: Would simplify code generation
1. **Explicit relinearization key API**: Clear way to specify evaluation keys
1. **Dimension tracking**: API to query/set ciphertext dimensions
1. **Cost model documentation**: Detailed explanation of what costs are being
   modeled

### Potential Collaboration Areas

1. **Standardizing FHE cost modeling APIs**: Could SimFHE's API become a
   standard interface?
1. **Compiler-friendly features**: What changes would make SimFHE easier to
   target from compilers?
1. **Validation suite**: Test cases to verify correct SimFHE usage from code
   generators
1. **Performance accuracy**: Validating SimFHE estimates against real FHE
   execution

## Appendix

### Full Generated Code Example

See `simfhe_generated.py` for a complete example of HEIR-generated SimFHE code.

### Build/Run Instructions

1. Use Alexander Viand's fork or add "generated" to `DECORATION_LIST` in
   upstream SimFHE
1. Generate SimFHE code:
   ```bash
   heir-opt --secretize --mlir-to-ckks input.mlir | heir-translate --emit-simfhe > generated.py
   ```
1. Place `generated.py` in SimFHE directory
1. Run: `python generated.py`

### Contact

For HEIR-specific questions: https://github.com/google/heir For integration
issues: Alexander Viand (fork maintainer)
