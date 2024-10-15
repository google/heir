"""Macros to streamline the generation of HEIR transform tablegen-generated
files."""

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

def add_heir_transforms(
        td_file = "Passes.td",
        pass_name = None,
        header_filename = "Passes.h.inc",
        registration_name = None,
        doc_filename = None,
        generated_target_name = "pass_inc_gen",
        deps = None):
    """A HEIR standard way to invoke tablegen for transforms.

    This macro calls `gentbl_cc_library` with appropriate parameters, generating:

    - A header file, called `Passes.h.inc` by default.
    - A documentation file, called `${pass_name}Passes.md` by default.
    - A bazel target containing the generated headers, called `pass_inc_gen` by
      default.

    Args:
      td_file: A string containing the path to the tablegen file to process. Uses
        "Passes.td" by default. This file should contain a `def` inheriting from
        `Pass` for each pass defined. Multiple passes may be defined in the same
        file.
      pass_name: A string containing the name of the pass to generate.
        This produces:
          - A registration function called `register${pass_name}Passes`
          - A markdown file for the documentation called `${pass_name}Passes.md`
      header_filename: An override for the header filename `Passes.h.inc`.
      registration_name: An override for the registration function defined by
        pass_name: generated as `register${registration_name}Passes`.
      doc_filename: An override for the documentation filename set by pass_name.
      generated_target_name: An override for the generated bazel target name.
      deps: A list of tablegen dependencies required to generate the target.
        Defaults to OpBaseTdFiles and PassBaseTdFiles.
    """
    if td_file == None:
        fail("td_file must be provided to add_heir_transforms.")

    if pass_name == None and registration_name == None and doc_filename == None:
        fail("pass_name must be provided to add_heir_transforms, or overrides for" +
             "registration_name and doc_filename must be provided.")

    _doc_filename = None
    pass_decls_name = None

    if pass_name != None:
        pass_decls_name = pass_name
        _doc_filename = pass_name + "Passes.md"

    if registration_name != None:
        pass_decls_name = registration_name

    if doc_filename != None:
        _doc_filename = doc_filename

    _deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ]
    if deps != None:
        _deps = deps

    gentbl_cc_library(
        name = generated_target_name,
        tbl_outs = [
            (
                [
                    "-gen-pass-decls",
                    "-name=" + pass_decls_name,
                ],
                header_filename,
            ),
            (
                ["-gen-pass-doc"],
                _doc_filename,
            ),
        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = td_file,
        deps = _deps,
    )
