# SimFHE Example Package Plan

## Goal

Create a comprehensive package demonstrating how HEIR's SimFHE backend works,
including:

- A simple MLIR program compilation to SimFHE
- The generated SimFHE Python code
- Actual SimFHE execution results
- Documentation of assumptions and limitations

## Components to Include

### 1. Example Program Selection

- Use a simple CKKS-compatible example that demonstrates key operations
- Create a custom example that:
  - Takes encrypted inputs
  - Performs basic arithmetic (add, mul)
  - Includes a rotation
  - Returns encrypted result
- Keep it simple enough to trace through manually

### 2. Full Pipeline Demonstration

Show the complete flow:

1. **Input**: High-level MLIR (secret dialect)
1. **Intermediate**: CKKS dialect after lowering
1. **Output**: Generated SimFHE Python code
1. **Comparison**: Clean OpenFHE code for reference
1. **Results**: SimFHE execution output

### 3. SimFHE Fork Documentation

- Clone Alexander Viand's fork: https://github.com/alexanderviand/SimFHE
- Document differences from upstream SimFHE
- Explain why the fork is needed
- Note any specific configuration required

### 4. Implementation Analysis

Document:

- Key assumptions in the SimFHE emitter
- Mapping of CKKS operations to SimFHE API
- Limitations and workarounds (e.g., sub → add)
- Parameter selection rationale

### 5. Execution Environment Setup

Provide:

- Installation instructions for SimFHE fork
- Required Python dependencies
- Directory structure setup
- Running the generated code

### 6. Results Interpretation

Explain:

- What SimFHE outputs mean
- Performance metrics provided
- How to interpret the results
- Caveats about accuracy

## Structure of Final Document

```
simfhe_example_package.md
├── Introduction
├── Example Program
│   ├── High-level MLIR
│   ├── CKKS MLIR
│   └── Reference OpenFHE code
├── SimFHE Backend
│   ├── Fork details
│   ├── Setup instructions
│   └── Assumptions & limitations
├── Generated Code
│   ├── Full SimFHE Python
│   └── Key sections explained
├── Execution
│   ├── Running the code
│   ├── Output interpretation
│   └── Performance results
└── Analysis & Recommendations
```

## Next Steps

1. Create a simple but representative MLIR example
1. Run it through the HEIR pipeline
1. Set up SimFHE fork and test generated code
1. Document all findings in a single comprehensive markdown file
