# SimFHE Output Interpretation Guide (Internal)

## Understanding SimFHE Output

When running HEIR-generated SimFHE code, you'll see output like this:

```
(17, 2, 6, FFTStyle.BSGS, CacheStyle.ALPHA, 40, 61) -> 19
(17, 2, 6, FFTStyle.BSGS, CacheStyle.ALPHA, 40, 61) -> 19
(17, 2, 6, FFTStyle.UNROLLED_HOISTED, CacheStyle.ALPHA, 40, 61) -> 19
(17, 2, 6, FFTStyle.UNROLLED_HOISTED, CacheStyle.ALPHA, 40, 61) -> 19
fn                 total ops    total mult    dram total    dram limb rd    dram limb wr    dram key rd
---------------  -----------  ------------  ------------  --------------  --------------  -------------
polynomial_eval      3.10811       1.3039       0.756941        0.209715        0.127795       0.41943
polynomial_eval      2.83705       1.20848      0.756941        0.209715        0.127795       0.41943
polynomial_eval      2.83705       1.20848      0.756941        0.209715        0.127795       0.41943
polynomial_eval      2.83705       1.20848      0.547226        0.209715        0.127795       0.209715
```

## Parameter Tuples

The parameter tuples represent different FHE scheme configurations:

- `(17, 2, 6, FFTStyle.BSGS, CacheStyle.ALPHA, 40, 61) -> 19`

Breaking this down:

1. **17**: Log of polynomial degree (logN)
1. **2**: Number of primes in RNS representation
1. **6**: Multiplicative depth
1. **FFTStyle**: NTT implementation style
   - `BSGS`: Baby-step Giant-step algorithm
   - `UNROLLED_HOISTED`: Unrolled loops with hoisted operations
1. **CacheStyle.ALPHA**: Cache management strategy
1. **40**: Log of first prime modulus
1. **61**: Log of scale
1. **-> 19**: Resulting security level

## Performance Metrics

The table shows various performance costs:

### Core Metrics

- **total ops**: Total operation count (in some normalized unit)
- **total mult**: Total multiplication operations
- **dram total**: Total DRAM access cost
- **dram limb rd**: DRAM reads for RNS limbs
- **dram limb wr**: DRAM writes for RNS limbs
- **dram key rd**: DRAM reads for evaluation keys

### Interpretation

- Lower values are better (less computational cost)
- Different parameter sets show trade-offs
- The last row often shows best parameters (lowest DRAM total)

## What SimFHE Models

SimFHE estimates:

1. **Computational costs**: Number of polynomial operations
1. **Memory access patterns**: DRAM read/write costs
1. **Key material access**: Evaluation key usage
1. **FFT/NTT costs**: Based on implementation style

## Limitations

1. **Relative costs**: Values are normalized, not absolute time
1. **Hardware assumptions**: Based on specific accelerator model
1. **Simplified model**: Doesn't capture all FHE implementation details
1. **No network costs**: Assumes single-machine execution

## Using Results

1. **Compare implementations**: Look at relative costs between different
   approaches
1. **Identify bottlenecks**: High DRAM costs suggest memory-bound computation
1. **Parameter selection**: Choose parameters with lowest total cost
1. **Optimization targets**: Focus on operations with highest costs

## Common Patterns

1. **Multiplication dominates**: `total mult` often correlates with total cost
1. **Key reads expensive**: Relinearization shows in `dram key rd`
1. **FFT style matters**: UNROLLED_HOISTED usually faster than BSGS
1. **Parameter trade-offs**: Larger logN increases security but also cost
