# Cornami FHE Compiler

Cornami FHE Compiler is setup as a exit dialect and codegen from one of
**CGGI**, **TfheRustBool**, **Combinational** from **HEIR** project.

We plan to do this using a custom dialect **SCIFRBool** for boolean operations
first, then another dialect **SCIFR** for compiling **CKKS** based real number
dialects.

The Cornami backend focused tools are available by default.

# heir-opt

## Estimations

1. --cggi-tigris-estimator - Estimate resources for the CGGI program on Cornami
   Tigris chip(s)
1. --ckks-tigris-estimator - Estimate resources for the CKKS program on Cornami
   Tigris chip(s)

## Conversions

1. ```
    --cggi-to-scifrbool                              -   Lower `cggi` to `scifrbool` dialect.
   ```

# heir-translate

## Codegen

1. --emit-scifrbool option

<!-- mdformat global-off -->
