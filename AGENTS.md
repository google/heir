This document is intended for AI-powered coding assistants working on the HEIR
repository.

## Repository Overview

- **Name**: HEIR (Homomorphic Encryption Intermediate Representation)
- **Website & Docs**: https://heir.dev
- **Build System**: Bazel
- **Key Directories**:
  - `lib/`: Contains the (C++) code for the core compiler
  - `frontend/`: Python API, examples, and unit tests
  - `tests/`: MLIR and Python test cases, MLIR lit tests
  - `scripts/`: Utility scripts (e.g., lit-to-Bazel converter)
  - `docs/`: Hugo site content for user & developer documentation

### File System Operations

- **Avoid** slow filesystem scans: do not use `ls -R`, `find`, or `grep` on
  large paths. Instead, use ripgrep when it is available on the machine.

### Pre-commit and Testing

- After applying patches, run style linters on modified files:
  ```bash
  pre-commit run --all-files
  ```
- If the repository’s pre-commit setup is broken (failing on untouched lines),
  inform the user rather than fixing unrelated errors.
- To verify functionality, run the full test suite:
  ```bash
  bazel test @heir//...       # MLIR passes, Python scripts, lit tests
  pytest frontend/ tests/     # Python unit tests
  ```

## Coding Guidelines

When making changes, follow these practices:

1. **Fix the root cause** rather than applying surface-level patches.
1. **Minimize scope**: keep changes focused on the user’s request.
1. **Maintain style**: follow existing code conventions and formatting.
1. **No unrelated changes**: avoid touching unrelated files or tests.
1. **Avoid inline comments** unless absolutely necessary for clarity.
1. **Update documentation** (e.g., `README.md`, `docs/`) if behavior or APIs
   change.
