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


class CLI:
    def __init__(self):
        git_root = pathlib.Path(__file__).parent.parent
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
            pass_name: The CPP class name and directory name for the conversion pass, e.g., BGVToPolynomial
            source_dialect_name: The source dialect's CPP class name prefix and directory name, e.g., CGGI (for CGGIDialect)
            source_dialect_namespace: The source dialect's CPP namespace, e.g., tfhe_rust for TfheRustDialect
            source_dialect_mnemonic: The source dialect's mnemonic, e.g., cggi
            target_dialect_name: The target dialect's CPP class name prefix and directory name, e.g., CGGI (for CGGIDialect)
            target_dialect_namespace: The target dialect's CPP namespace, e.g., tfhe_rust for TfheRustDialect
            target_dialect_mnemonic: The target dialect's mnemonic, e.g., cggi
            force: If True, overwrite existing files. If False, raise an error if any files already exist.
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

        include_path = self.root / "include" / "Conversion" / pass_name
        lib_path = self.root / "lib" / "Conversion" / pass_name

        if not force and (os.path.isdir(include_path) or os.path.isdir(lib_path)):
            raise ValueError(
                f"Conversion pass directories already exist at {include_path} or {lib_path}"
            )

        try:
            print(f"Creating dirs:\n  {include_path}\n  {lib_path}")
            try:
                os.mkdir(include_path)
                os.mkdir(lib_path)
            except FileExistsError:
                if force:
                    shutil.rmtree(include_path)
                    shutil.rmtree(lib_path)
                    os.mkdir(include_path)
                    os.mkdir(lib_path)
                else:
                    raise

            # Copy and render jinja templates.
            templates_path = self.root / "templates" / "Conversion"
            shutil.copy(
                templates_path / "include" / "BUILD.jinja", include_path / "BUILD"
            )
            shutil.copy(
                templates_path / "include" / "ConversionPass.h.jinja",
                include_path / f"{pass_name}.h",
            )
            shutil.copy(
                templates_path / "include" / "ConversionPass.td.jinja",
                include_path / f"{pass_name}.td",
            )
            shutil.copy(templates_path / "lib" / "BUILD.jinja", lib_path / "BUILD")
            shutil.copy(
                templates_path / "lib" / "ConversionPass.cpp.jinja",
                lib_path / f"{pass_name}.cpp",
            )
            render_all(
                include_path,
                pass_name=pass_name,
                source_dialect_name=source_dialect_name,
                source_dialect_namespace=source_dialect_namespace,
                source_dialect_mnemonic=source_dialect_mnemonic,
                target_dialect_name=target_dialect_name,
                target_dialect_namespace=target_dialect_namespace,
                target_dialect_mnemonic=target_dialect_mnemonic,
            )
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
            shutil.rmtree(include_path)
            shutil.rmtree(lib_path)
            raise


if __name__ == "__main__":
    fire.Fire(CLI)
