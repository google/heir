---
title: Noise Analysis
weight: 9
---

Homomorphic Encryption (HE) schemes based on Learning-With-Errors (LWE) and
Ring-LWE naturally need to deal with *noises*. HE compilers, in particular, need
to understand the noise behavior to ensure correctness and security while
pursuing efficiency and optimizaiton.

The noise analysis in HEIR has the following central task: Given an HE circuit,
analyse the noise growth for each operation. HEIR then uses noise analysis for
parameter selection, but the details of that are beyond the scope of this
document.

Noise analysis and parameter generation are still under active researching and
HEIR does not have a one-size-fits-all solution for now. Noise analyses and
(especially) parameter generation in HEIR should be viewed as experimental.
*There is no guarantee that they are correct or secure* and the HEIR authors do
not take responsibility. Please consult experts before putting them into
production.

## Two Flavors of Noise Analysis

Each HE ciphertext contains *noise*. A noise analysis determines a *bound* on
the noise and tracks its evolution after each HE operation. The noise should not
exceed certain bounds imposed by HE schemes.

There are two flavors of noise analyses: worst-case and average-case. Worst-case
noise analyses always track the bound, while some average-case noise analyses
use intermediate quantity like the variance to track their evolution, and derive
a bound when needed.

Currently, worst-case methods are often too conservative, while average-case
methods often give underestimation.

## Noise Analysis Framework

HEIR implements noise analysis based on the `DataFlowFramework` in MLIR.

In the `DataFlowFramework`, the main function of an `Analysis` is
`visitOperation`, where it determines the `AnalysisState` for each SSA `Value`.
Usually it computes a transfer function deriving the `AnalysisState` for each
operation result based on the states of the operation's operands.

As there are various HE schemes in HEIR, the detailed transfer function is
defined by a `NoiseModel` class, which parameterizes the `NoiseAnalysis`.

The `AnalysisState`, depending on whether we are using worst-case noise model or
average-case, could be interpreted as the bound or the variance.

A typical way to use noise analysis:

```cpp
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"  // from @llvm-project

DataFlowSolver solver;
dataflow::loadBaselineAnalyses(solver);
// load other dependent analyses

// schemeParam and model determined by other methods
solver.load<NoiseAnalysis<NoiseModel>>(schemeParam, model);

// run the analysis on the op
solver.initializeAndRun(op)
```

## Implemented Noise Models

See the [Passes](/docs/passes) page for details. Example passes include
`generate-param-bgv` and `validate-noise`.

<!-- mdformat global-off -->
