# HEIR -> Intel HERACLES Integration

This document (which should not be added to the repo, but could be made public as it does not contain IC/ITS information) describes the steps required for integration of the MLIR-based HEIR toolchain with the Intel HERACLES FHE Accelerator and associated SDK.

## Path A: OpenFHE Integration
HEIR can produce C++ OpenFHE code from high-level programs, applying simplifications such as automated batching (cf. Viand et al., Usenix'23 - which has been fully integrated into HEIR).
The resulting C++ source file uses the OpenFHE FHE API just as any hand-written OpenFHE application would.
As a result, if compiled against a special augmented version of OpenFHE, it can be used to generate traces suitable for conversion to Intel Heracles via the SDK, once the SDK's OpenFHE->Heracles toolchain becomes available.

## Stage 1: Polynomial to p-ISA Integration
HEIR's Intermediate Representation (IR) for polynomial arithmetic (now also upstream in LLVM/MLIR) can be translated to  the pISA used by the SDK's functional simulator and assembler.
Both HEIR's polynomial IR and pISA are infinite register abstraction without memory management concerns.
However, whereas pISA freely mixes coefficient-representation and evaluation/ntt-representation,
HEIR is more strict, with coefficient-representation modelled with the `polynomial` type, while ntt-representations are modelled as modular arithmetic over tensors of integers (currently using `arith_ext` dialect, but that is likely to be renamed to `modarith` or similar).

### pISA Instructions and HEIR equivalents.
* Common Preamble for HEIR, where `Q` is the modulus from the pISA instructions and `R` is a primitive 2n-th root of unity of `Q`, with `N` the polynomial degree:
    ```llvm
    #ideal = #polynomial.int_polynomial<1 + x**N>
    #ring = #polynomial.ring<coefficientType = i32, coefficientModulus = Q : i32, polynomialModulus=#ideal, root=R>
    ```
* `padd d,c1,c2,q`:
    ```llvm
    %d = polynomial.add %c1, %2 : !polynomial.polynomial<#ring>
    ```
* `psub d,c1,c2,q`: (**Q: Operand order?**)
    ```llvm
    %d = polynomial.sub %c1, %2 : !polynomial.polynomial<#ring>
    ```
* `pmul d,c1,c2,q`:
    Note: `pmul` is elementwise multiplication, while HEIR's `polynomial.mul` is full polynomial multiplication.
    As a result, this piSA operation is *not* equivalent to
    ```llvm
    %d = polynomial.mul %c1, %c2 : !polynomial.polynomial<#ring>
    ```
    but rather
    ```llvm
    %0 = polynomial.ntt %c1 : !polynomial.polynomial<#ring> -> tensor<Nxi32, #ring>
    %1 = polynomial.ntt %c2 : !polynomial.polynomial<#ring> -> tensor<Nxi32, #ring>
    %2 = arith_ext.mul %0, %1 {modulus = Q : i32} : tensor<Nxi32, #ring>
    %7 = polynomial.intt %6 : tensor<4xi5, #ring> -> !polynomial.polynomial<#ring>
    ```
    We can go from the top representation to the bottom with the `--convert-polynomial-mul-to-ntt` pass.
* `pmuli d,c,imm,q`: (**Q: type of imm? (bitwidth? signedness? modulo?)**)
    ```llvm
    %i = arith.constant IMM : i32
    %d = polynomial.mul_scalar %c, %i : !polynomial.polynomial<#ring>, i32
    ```
* `pmac d,c1,c2,q`:  (**Q: Does d get overwritten here? So not SSA?**)
    Note: omitting `ntt`/`intt` here for clearer exposition
    Note: pISA seems to allow overriding `d`, which is not possible in SSA representations like MLIR.
    ```llvm
    %d_new = arith_ext.mac %c1, %c2, %d {modulus = q : i32} : tensor<Nxi32, #ring>
    ```
* `pmaci d,c,imm,q`:
    ```llvm
    %i = arith.constant dense<IMM> : tensor<Nxi32, #ring>
    %d_new = arith_ext.mac %c, %i, %d {modulus = q : i32} : tensor<Nxi32, #ring>
    ```
* `pntt d,c,q`:
    ```llvm
    %d = polynomimal.ntt $c :  polynomial.polynomial<#ring> -> tensor<Nxi32, #ring>
    ```
* `pintt d, c, q`:
    ```llvm
    %d = polynomial.intt %c : tensor<4xi5, #ring> -> !polynomial.polynomial<#ring>
    ```

### HEIR -> pISA Mapping

* `polynomimal.mul`: Not supported, use `--convert-polynomial-mul-to-ntt` first!

* `polynomial.add`, `polynomial.ntt`, `polynomial.intt` are trivial 1:1 mappings

* `arith_ext.add`, `arith_ext.mul`, `arith_ext.mac` are trivial mappings

* `arith.constant` is converted to an additional input.
  TODO: in the future, check if we can do immediate stuff first!

*
