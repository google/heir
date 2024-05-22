"""A CLI for generating boilerplate MLIR code in HEIR."""
import os
import pathlib
import shutil

import fire
import jinja2


def render_all(path: pathlib.Path, **args):
  env = jinja2.Environment(loader=jinja2.FileSystemLoader(path))
  for template_filename in os.listdir(path):
    template = env.get_template(template_filename)
    content = template.render(**args)
    with open(path / template_filename, mode="w") as outfile:
      outfile.write(content)
      print(f"Rendered template for {path / template_filename}")


def try_create_dirs(lib_path, force=False):
  print(f"Creating dirs:\n  {lib_path}")
  try:
    os.makedirs(lib_path)
  except FileExistsError:
    if force:
      shutil.rmtree(lib_path)
      os.mkdir(lib_path)
    else:
      raise


def copy_all(filepath_mapping):
  for src, dest in filepath_mapping.items():
    shutil.copy(src, dest)


class CLI:
  """A helper CLI for generating boilerplate MLIR code in HEIR.

  Available subcommands:

    new_conversion_pass: Create a conversion pass from one dialect to another.
    new_dialect_transform: Create a pass for a dialect-specific transform.
    new_dialect: Create a new dialect.
    new_transform: Create a pass for a non-dialect-specific transform.

  To see the help for a subcommand, run

    python scripts/templates/templates.py <subcommand> --help
  """

  def __init__(self):
    git_root = pathlib.Path(__file__).parent.parent.parent
    if not os.path.isdir(git_root / ".git"):
      raise RuntimeError(f"Could not find git root, looked at {git_root}")
    self.root = git_root

  def new_conversion_pass(
      self,
      pass_name: str = None,
      source_dialect_name: str = None,
      source_dialect_namespace: str = None,
      source_dialect_mnemonic: str = None,
      target_dialect_name: str = None,
      target_dialect_namespace: str = None,
      target_dialect_mnemonic: str = None,
      force: bool = False,
  ):
    """Create a new conversion pass.

        Args:
          pass_name: The CPP class name and directory name for the conversion
            pass, e.g., BGVToPolynomial
          source_dialect_name: The source dialect's CPP class name prefix and
            directory name, e.g., CGGI (for CGGIDialect)
          source_dialect_namespace: The source dialect's CPP namespace, e.g.,
            tfhe_rust for TfheRustDialect
          source_dialect_mnemonic: The source dialect's mnemonic, e.g., cggi
          target_dialect_name: The target dialect's CPP class name prefix and
            directory name, e.g., CGGI (for CGGIDialect)
          target_dialect_namespace: The target dialect's CPP namespace, e.g.,
            tfhe_rust for TfheRustDialect
          target_dialect_mnemonic: The target dialect's mnemonic, e.g., cggi
          force: If True, overwrite existing files. If False, raise an error if
            any files already exist.
        """
    if not source_dialect_name:
      raise ValueError("source_dialect_name must be provided")
    if not target_dialect_name:
      raise ValueError("target_dialect_name must be provided")

    if not pass_name:
      pass_name = f"{source_dialect_name}To{target_dialect_name}"

    # These defaults could be smarter: look up the name in the actual
    # tablegen for the dialect or quit if it can't be found
    if not source_dialect_mnemonic:
      source_dialect_mnemonic = source_dialect_name.lower()

    if not source_dialect_namespace:
      source_dialect_namespace = source_dialect_mnemonic

    if not target_dialect_mnemonic:
      target_dialect_mnemonic = target_dialect_name.lower()

    if not target_dialect_namespace:
      target_dialect_namespace = target_dialect_mnemonic

    lib_path = self.root / "lib" / "Conversion" / pass_name

    if not force and os.path.isdir(lib_path):
      raise ValueError(
          f"Conversion pass directories already exist at {lib_path}")

    templates_path = self.root / "scripts" / "templates" / "Conversion"
    templ_lib = templates_path / "lib"
    path_mapping = {
        templ_lib / "ConversionPass.h.jinja": lib_path / f"{pass_name}.h",
        templ_lib / "ConversionPass.td.jinja": lib_path / f"{pass_name}.td",
        templ_lib / "BUILD.jinja": lib_path / "BUILD",
        templ_lib / "ConversionPass.cpp.jinja": lib_path / f"{pass_name}.cpp",
    }

    try:
      try_create_dirs(lib_path, force)
      copy_all(path_mapping)

      render_all(
          lib_path,
          pass_name=pass_name,
          source_dialect_name=source_dialect_name,
          source_dialect_namespace=source_dialect_namespace,
          source_dialect_mnemonic=source_dialect_mnemonic,
          target_dialect_name=target_dialect_name,
          target_dialect_namespace=target_dialect_namespace,
          target_dialect_mnemonic=target_dialect_mnemonic,
      )
    except:
      print("Hit unrecoverable error, cleaning up")
      shutil.rmtree(lib_path)
      raise

  def new_dialect_transform(
      self,
      pass_name: str = None,
      pass_flag: str = None,
      dialect_name: str = None,
      dialect_namespace: str = None,
      force: bool = False,
  ):
    """Create a new pass for a dialect-specific transform.

        Args:
          pass_name: The CPP class name for the pass, e.g., ForgetSecrets.
          pass_flag: The CLI flag to use for the pass (optional).
          dialect_name: The dialect's CPP class name prefix and directory name,
            e.g., CGGI (for CGGIDialect).
          dialect_namespace: The dialect's CPP namespace, e.g., tfhe_rust for
            TfheRustDialect.
          force: If True, overwrite existing files. If False, raise an error if
            any files already exist.
        """
    if not pass_name:
      raise ValueError("pass_name must be provided")
    if not dialect_name:
      raise ValueError("dialect_name must be provided")

    if not pass_flag:
      pass_flag = f"{dialect_name.lower()}-{pass_name.lower()}"

    # Default could be smarter: look up the name in the actual tablegen for the
    # dialect or quit if it can't be found
    if not dialect_namespace:
      dialect_namespace = dialect_name.lower()

    lib_path = self.root / "lib" / "Dialect" / dialect_name / "Transforms"

    if not force and os.path.isdir(lib_path):
      raise ValueError(
          f"Pass directories already exist at {lib_path}")

    templates_path = self.root / "scripts" / "templates" / "DialectTransforms"
    templ_lib = templates_path / "lib"
    path_mapping = {
        templ_lib / "BUILD.jinja": lib_path / "BUILD",
        templ_lib / "Pass.cpp.jinja": lib_path / f"{pass_name}.cpp",
        templ_lib / "Pass.h.jinja": lib_path / f"{pass_name}.h",
        templ_lib / "Passes.h.jinja": lib_path / "Passes.h",
        templ_lib / "Passes.td.jinja": lib_path / "Passes.td",
    }

    try:
      try_create_dirs(lib_path, force)
      copy_all(path_mapping)
      render_all(
          lib_path,
          pass_name=pass_name,
          pass_flag=pass_flag,
          dialect_name=dialect_name,
          dialect_namespace=dialect_namespace,
      )
    except:
      print("Hit unrecoverable error, cleaning up")
      shutil.rmtree(lib_path)
      raise

  def new_transform(
      self,
      pass_name: str = None,
      pass_flag: str = None,
      force: bool = False,
  ):
    """Create a new pass for a dialect-specific transform.

        Args:
          pass_name: The CPP class name for the pass, e.g., ForgetSecrets.
          pass_flag: The CLI flag to use for the pass (optional).
          force: If True, overwrite existing files. If False, raise an error if
            any files already exist.
        """
    if not pass_name:
      raise ValueError("pass_name must be provided")

    if not pass_flag:
      pass_name = f"{pass_name.lower()}"

    lib_path = self.root / "lib" / "Transforms" / pass_name

    if not force and os.path.isdir(lib_path):
      raise ValueError(
          f"Pass directories already exist at {lib_path}")

    templates_path = self.root / "scripts" / "templates" / "Transforms"
    templ_lib = templates_path / "lib"
    path_mapping = {
        templ_lib / "BUILD.jinja": lib_path / "BUILD",
        templ_lib / "Pass.cpp.jinja": lib_path / f"{pass_name}.cpp",
        templ_lib / "Pass.h.jinja": lib_path / f"{pass_name}.h",
        templ_lib / "Pass.td.jinja": lib_path / f"{pass_name}.td",
    }

    try:
      try_create_dirs(lib_path, force)
      copy_all(path_mapping)
      render_all(
          lib_path,
          pass_name=pass_name,
          pass_flag=pass_flag,
      )
    except:
      print("Hit unrecoverable error, cleaning up")
      shutil.rmtree(lib_path)
      raise

  def new_dialect(
      self,
      dialect_name: str = None,
      dialect_namespace: str = None,
      enable_attributes: bool = True,
      enable_types: bool = True,
      enable_ops: bool = True,
      force: bool = False,
  ):
    """Create a new dialect.

    Args:
      dialect_name: The dialect's CPP class name prefix and directory name,
        e.g., CGGI (for CGGIDialect).
      dialect_namespace: The dialect's CPP namespace, e.g., tfhe_rust for
        TfheRustDialect.
      enable_attributes: Generate a separate tablegen and includes for
        attributes.
      enable_types: Generate a separate tablegen and includes for types.
      enable_ops: Generate a separate tablegen and includes for ops.
      force: If True, overwrite existing files. If False, raise an error if
        any files already exist.
    """
    if not dialect_name:
      raise ValueError("dialect_name must be provided")

    if not dialect_namespace:
      dialect_namespace = dialect_name.lower()

    lib_path = self.root / "lib" / "Dialect" / dialect_name / "IR"

    if not force and os.path.isdir(lib_path):
      raise ValueError(
          f"Dialect directories already exist at {lib_path}")

    templates_path = self.root / "scripts" / "templates" / "Dialect"
    templ_lib = templates_path / "lib"
    path_mapping = {
        templ_lib / "BUILD.jinja": lib_path / "BUILD",
        templ_lib / "Dialect.cpp.jinja": lib_path / f"{dialect_name}Dialect.cpp",
        templ_lib / "Dialect.h.jinja": lib_path / f"{dialect_name}Dialect.h",
        templ_lib / "Dialect.td.jinja": lib_path / f"{dialect_name}Dialect.td",
    }

    if enable_attributes:
      path_mapping.update({
          templ_lib
          / "Attributes.h.jinja": lib_path / f"{dialect_name}Attributes.h",
          templ_lib
          / "Attributes.td.jinja": (
              lib_path / f"{dialect_name}Attributes.td"
          ),
          templ_lib
          / "Attributes.cpp.jinja": lib_path / f"{dialect_name}Attributes.cpp",
      })

    if enable_types:
      path_mapping.update({
          templ_lib
          / "Types.h.jinja": lib_path / f"{dialect_name}Types.h",
          templ_lib
          / "Types.td.jinja": lib_path / f"{dialect_name}Types.td",
          templ_lib / "Types.cpp.jinja": lib_path / f"{dialect_name}Types.cpp",
      })

    if enable_ops:
      path_mapping.update({
          templ_lib / "Ops.h.jinja": lib_path / f"{dialect_name}Ops.h",
          templ_lib
          / "Ops.td.jinja": lib_path / f"{dialect_name}Ops.td",
          templ_lib / "Ops.cpp.jinja": lib_path / f"{dialect_name}Ops.cpp",
      })

    try:
      try_create_dirs(lib_path, force)
      copy_all(path_mapping)
      render_all(
          lib_path,
          dialect_name=dialect_name,
          dialect_namespace=dialect_namespace,
          enable_attributes=enable_attributes,
          enable_types=enable_types,
          enable_ops=enable_ops,
      )
    except:
      print("Hit unrecoverable error, cleaning up")
      shutil.rmtree(lib_path)
      raise


if __name__ == "__main__":
  fire.Fire(CLI)
