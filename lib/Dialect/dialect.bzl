"""Macros to streamline the generation of HEIR dialect tablegen-generated
files."""

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

def add_heir_dialect_library(
        name = None,
        dialect = None,
        kind = None,
        td_file = None,
        deps = []):
    """Generates a .inc library for a HEIR dialect.

    Args:
        name: the name of the target to generate.
        dialect: The name of the dialect.
        kind: The kind of tablegen file to generate.
        td_file: The .td file to use for tablegen.
        deps: The dependencies of the generated target.
    """
    if name == None:
        fail("name must be provided to add_heir_dialect_library.")
    if dialect == None:
        fail("dialect must be provided to add_heir_dialect_library.")
    if td_file == None:
        fail("td_file must be provided to add_heir_dialect_library.")

    _tblgen_command_prefix = "-gen-"
    _tblgen_command_infix = None
    _tblgen_command_suffix_decls = "-decls"
    _tblgen_command_suffix_defs = "-defs"
    _tblgen_command_suffix_doc = "-doc"

    _file_name_prefix = dialect
    _file_name_infix = None

    if kind == "dialect":
        _tblgen_command_infix = "dialect"
        _file_name_infix = "Dialect"
    elif kind == "attribute":
        _tblgen_command_infix = "attrdef"
        _file_name_infix = "Attributes"
    elif kind == "enum":
        _tblgen_command_infix = "enum"
        _file_name_infix = "Enums"
    elif kind == "type":
        _tblgen_command_infix = "typedef"
        _file_name_infix = "Types"
    elif kind == "op":
        _tblgen_command_infix = "op"
        _file_name_infix = "Ops"
    elif kind == "type_interface":
        _tblgen_command_infix = "type-interface"
        _tblgen_command_suffix_doc = "-docs"
        _file_name_infix = "TypeInterfaces"
    else:
        fail("kind must be provided to add_heir_dialect_library.")

    _header_inc_file = _file_name_prefix + _file_name_infix + ".h.inc"
    _cpp_inc_file = _file_name_prefix + _file_name_infix + ".cpp.inc"
    _doc_inc_file = _file_name_prefix + _file_name_infix + ".md"

    _tblgen_command_decls = [_tblgen_command_prefix + _tblgen_command_infix + _tblgen_command_suffix_decls]
    _tblgen_command_defs = [_tblgen_command_prefix + _tblgen_command_infix + _tblgen_command_suffix_defs]
    _tblgen_command_doc = [_tblgen_command_prefix + _tblgen_command_infix + _tblgen_command_suffix_doc]

    gentbl_cc_library(
        name = name,
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
        td_file = td_file,
        deps = deps,
    )
