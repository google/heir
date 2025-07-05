# HEIR: Homomorphic Encryption Intermediate Representation

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/google/heir/build_and_test.yml)
![GitHub Contributors](https://img.shields.io/github/contributors/google/heir)
![GitHub Discussions](https://img.shields.io/github/discussions/google/heir)
![GitHub License](https://img.shields.io/github/license/google/heir)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/google/heir/badge)](https://securityscorecards.dev/viewer/?uri=github.com/google/heir)

An MLIR-based toolchain for
[homomorphic encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption)
compilers. Read the docs at [the HEIR website](https://heir.dev).

For more information on MLIR, see the [MLIR homepage](https://mlir.llvm.org/).

## Quickstart (Python)

Pip install the `heir_py` package

```bash
pip install heir_py
```

Then run an example:

```python
from heir import compile
from heir.mlir import F32, I16, I64, Secret

@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def func(x: Secret[I16], y: Secret[I16]):
    sum = x + y
    diff = x - y
    mul = x * y
    expression = sum * diff + mul
    deadcode = expression * mul
    return expression

foo.setup()
enc_a = foo.encrypt_a(7)
enc_b = foo.encrypt_b(8)
result_enc = foo.eval(enc_a, enc_b)
result = foo.decrypt_result(result_enc)

print(
  f"Expected result for `func`: {func.original(7,8)}, FHE result:"
  f" {result}"
)
```

This will compile the function above using the BGV scheme to machine code via
the [OpenFHE](https://openfhe-development.readthedocs.io/en/latest/) backend.
Then calling the function will encrypt the inputs, run the function, and return
the decrypted result. The function call `foo(7, 8)` runs the entire
encrypt-run-decrypt flow for ease of testing.

## Quickstart (heir-opt, heir-translate)

The python package `heir_py` ships with the `heir-opt` and `heir-translate`. If
you install via `virtualenv`, the binaries will be in your `venv/bin`.

```bash
venv/bin/heir-opt --help
```

## Supported backends and schemes

| Backend Library | BGV | BFV | CKKS | CGGI |
| --------------- | --- | --- | ---- | ---- |
| OpenFHE         | ✅  | ✅  | ✅   | ❌   |
| Lattigo         | ✅  | ✅  | ✅   | ❌   |
| tfhe-rs         | ❌  | ❌  | ❌   | ✅   |
| Jaxite          | ❌  | ❌  | ❌   | ✅   |

Note some backends do not support all schemes.

## Contributing

There are many ways to contribute to HEIR:

- Come to our [monthly meetings](https://heir.dev/community/) to discuss active
  work on HEIR and future project directions. The meetings are recorded and
  posted to our [blog](https://heir.dev/blog/) and
  [YouTube channel](https://www.youtube.com/@HEIRCompiler).
- Come to our [weekly office hours](https://heir.dev/community/) for informal
  discussions and debugging help.
- Ask questions or discuss feature ideas in the `#heir` channel on
  [the FHE.org discord](https://discord.fhe.org/).
- Work on an issue marked
  ["good first issue"](https://github.com/google/heir/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)
  or browse issues [labeled by topic](https://github.com/google/heir/labels).
- Help us understand new FHE research: either
  - Read a paper tagged under
    [research synthesis](https://github.com/google/heir/labels/research%20synthesis)
    and summarize the novel techniques that could be ported to HEIR.
  - File new issues under
    [research synthesis](https://github.com/google/heir/labels/research%20synthesis)
    to alert us of papers that should be investigated and incorporated into
    HEIR.

## Citations

The HEIR project can be cited in in academic work through following entry:

```text
@Misc{HEIR,
  title={{HEIR: Homomorphic Encryption Intermediate Representation}},
  author={HEIR Contributors},
  year={2023},
  note={\url{https://github.com/google/heir}},
}
```

## Support disclaimer

This is not an officially supported Google product.
