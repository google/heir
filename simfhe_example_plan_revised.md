# SimFHE Example Package Plan (Revised)

## Goal

Create a package for SimFHE maintainers demonstrating HEIR's SimFHE backend
integration, focusing on:

- Correctness of SimFHE API usage
- Integration assumptions and design decisions
- Potential areas for SimFHE improvement based on HEIR's needs

## Components to Include

### 1. Example Program

- Simple but representative CKKS computation
- Shows variety of operations (add, mul, rotate, relinearize)
- Clear mapping from FHE operations to SimFHE API calls

### 2. Fork Changes Summary

Brief overview of Alexander Viand's fork modifications:

- What was changed and why
- Whether these changes could be upstreamed
- Any HEIR-specific workarounds

### 3. Generated Code Analysis

For SimFHE maintainer review:

- How HEIR maps CKKS ops to SimFHE API
- Parameter passing conventions
- Any assumptions about SimFHE behavior
- Areas where SimFHE API was unclear/undocumented

### 4. Integration Challenges

Document for SimFHE team:

- Missing operations (e.g., subtraction)
- API limitations encountered
- Performance counter usage patterns
- Scheme parameter handling

### 5. Questions for SimFHE Team

- Is our usage of PerfCounter correct?
- Are we using the evaluator API as intended?
- What's the recommended way to handle relinearization keys?
- Any planned API changes we should know about?

## What SimFHE Maintainers Might Want to Know

1. **API Usage Patterns**

   - How a compiler (vs human) uses SimFHE
   - Automated code generation requirements
   - Type safety and parameter validation needs

1. **Performance Modeling**

   - What metrics HEIR users need
   - Accuracy requirements
   - Bootstrapping cost modeling

1. **Future Compatibility**

   - API stability requirements
   - Version compatibility needs
   - Feature requests from HEIR perspective

1. **Testing & Validation**

   - How to verify correct SimFHE usage
   - Test cases that would help HEIR
   - Regression testing considerations

## Structure of Final Document

```
simfhe_heir_integration.md
├── Executive Summary
│   └── Quick overview of HEIR's SimFHE usage
├── Integration Overview
│   ├── What HEIR is and why it uses SimFHE
│   ├── Fork changes summary
│   └── Current integration status
├── Example: CKKS Computation
│   ├── Original MLIR
│   ├── Generated SimFHE Python
│   └── Operation mapping table
├── Technical Details
│   ├── API usage patterns
│   ├── Parameter handling
│   ├── Workarounds & assumptions
│   └── Areas of uncertainty
├── Questions & Feedback
│   ├── API clarifications needed
│   ├── Feature requests
│   └── Potential collaboration areas
└── Appendix
    ├── Full generated code
    └── Build/run instructions
```

## Additional Files

1. `simfhe_output_interpretation.md` - For internal use, explaining SimFHE
   outputs
1. `fork_changes_detailed.md` - Detailed diff analysis of the fork

## Key Points to Emphasize for SimFHE Maintainers

- We're using SimFHE for cost modeling, not execution
- Generated code patterns vs hand-written
- Compiler requirements vs interactive use
- Documentation gaps we encountered
- Potential for standardizing FHE cost modeling APIs
