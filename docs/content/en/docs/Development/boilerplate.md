---
title: Boilerplate tools
weight: 40
---

The script `scripts/templates/templates.py` contains commands for generating new
dialects and transforms, filling in most of the boilerplate Tablegen and C++.
These commands do **not** add the code needed to register the new passes or
dialects in `heir-opt`.

These should be used when the tablegen files containing existing pass
definitions in the expected filepaths are not already present. Otherwise, you
must modify the existing tablegen files directly.

Run `python scripts/templates/templates.py --help` and
`python scripts/templates/templates.py <subcommand> --help` for the available
commands and options.

## Creating a New Pass

### General passes

If the pass does not operate from and to a specific dialect, use something
similar to:

```bash
python scripts/templates/templates.py new_transform \
--pass_name=ForgetSecrets \
--pass_flag=forget-secrets
```

### Dialect Transforms

To create a pass that operates within on a dialect, run a command similar to:

```bash
python scripts/templates/templates.py new_dialect_transform \
--pass_name=ForgetSecrets \
--pass_flag=forget-secrets \
--dialect_name=Secret \
--dialect_namespace=secret
```

### Conversion Pass

To create a new conversion pass, i.e., a pass that lowers from one dialect to
another, run a command similar to:

```bash
python scripts/templates/templates.py new_conversion_pass \
--source_dialect_name=CGGI \
--source_dialect_namespace=cggi \
--source_dialect_mnemonic=cggi \
--target_dialect_name=TfheRust \
--target_dialect_namespace=tfhe_rust \
--target_dialect_mnemonic=tfhe_rust
```

In order to build the resulting code, you must fix the labeled `FIXME`s in the
type converter and the op conversion patterns.

## Creating a New Dialect

To create a new dialect, run something like

```bash
python scripts/templates/templates.py new_dialect \
--dialet_name=TensorExt \
--dialect_namespace=tensor_ext \
--enable_attributes=False \
--enable_types=True \
--enable_ops=True
```

Note that all `--enable` flags are `True` by default, so if you know your
dialect will not have attributes or types, you have to explicitly disable those
options.
