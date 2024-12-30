"""Macros to streamline the generation of HEIR dialect tablegen-generated
files."""

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

def add_heir_dialect_library(
        dialect = None,
        kind = None,
        td_file = None,
        deps = []):
    """Generates a .inc library for a HEIR dialect.

    Args:
        dialect: The name of the dialect.
        kind: The kind of tablegen file to generate.
        td_file: The .td file to use for tablegen.
        deps: The dependencies of the generated target.
    """
    if dialect == None:
        fail("dialect must be provided to add_heir_dialect_library.")

    _target_name_prefix = None
    _target_name_suffix = "_inc_gen"

    _tblgen_command_prefix = "-gen-"
    _tblgen_command_infix = None
    _tblgen_command_suffix_decls = "-decls"
    _tblgen_command_suffix_defs = "-defs"
    _tblgen_command_suffix_doc = "-doc"

    _file_name_prefix = dialect
    _file_name_infix = None

    _dep_includes_dialect = True

    if kind == "dialect":
        _target_name_prefix = "dialect"
        _tblgen_command_infix = "dialect"
        _file_name_infix = "Dialect"
        _dep_includes_dialect = False
    elif kind == "attribute":
        _target_name_prefix = "attributes"
        _tblgen_command_infix = "attrdef"
        _file_name_infix = "Attributes"
    elif kind == "enum":
        _target_name_prefix = "enums"
        _tblgen_command_infix = "enum"
        _file_name_infix = "Enums"
    elif kind == "type":
        _target_name_prefix = "types"
        _tblgen_command_infix = "typedef"
        _file_name_infix = "Types"
    elif kind == "op":
        _target_name_prefix = "ops"
        _tblgen_command_infix = "op"
        _file_name_infix = "Ops"
    else:
        fail("kind must be provided to add_heir_dialect_library.")

    _td_file = td_file
    if _td_file == None:
        _td_file = _file_name_prefix + _file_name_infix + ".td"

    _target_name = _target_name_prefix + _target_name_suffix

    _header_inc_file = _file_name_prefix + _file_name_infix + ".h.inc"
    _cpp_inc_file = _file_name_prefix + _file_name_infix + ".cpp.inc"
    _doc_inc_file = _file_name_prefix + _file_name_infix + ".md"

    _tblgen_command_decls = [_tblgen_command_prefix + _tblgen_command_infix + _tblgen_command_suffix_decls]
    _tblgen_command_defs = [_tblgen_command_prefix + _tblgen_command_infix + _tblgen_command_suffix_defs]
    _tblgen_command_doc = [_tblgen_command_prefix + _tblgen_command_infix + _tblgen_command_suffix_doc]

    _deps = [":td_files"]
    if _dep_includes_dialect:
        _deps.append(":dialect_inc_gen")
    _deps = _deps + deps

    gentbl_cc_library(
        name = _target_name,
        tbl_outs = [
            (
                _tblgen_command_decls,
                _header_inc_file,
            ),
            (
                _tblgen_command_defs,
                _cpp_inc_file,
            ),
            (
                _tblgen_command_doc,
                _doc_inc_file,
            ),
        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = _td_file,
        deps = _deps,
    )
