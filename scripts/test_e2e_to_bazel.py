"""Tests for e2e_to_bazel.py."""

from contextlib import redirect_stdout
import io
from absl.testing import absltest
from scripts import e2e_to_bazel_lib

patch = absltest.mock.patch


class E2EToBazelTest(absltest.TestCase):
  """Tests for e2e_to_bazel script."""

  @patch("scripts.e2e_to_bazel_lib.run_blaze_query")
  @absltest.mock.patch.dict(
      "os.environ", {"BUILD_WORKSPACE_DIRECTORY": "/workspace"}
  )
  def test_e2e_to_bazel_with_target(self, mock_run_blaze_query):
    """Tests e2e_to_bazel with a target argument."""

    def side_effect(query_str, options=None):
      if options and "--output=xml" in options:
        return """<?xml version="1.1" encoding="UTF-8" standalone="no"?>
<query version="2">
    <rule class="heir_opt" name="//tests/Examples/openfhe/ckks/dot_product_8f:dot_product_8f_test_heir_opt">
        <list name="pass_flags">
            <string value="--annotate-module=backend=openfhe scheme=ckks"/>
            <string value="--mlir-to-ckks=ciphertext-degree=1024"/>
            <string value="--scheme-to-openfhe"/>
        </list>
        <label name="src" value="//tests/Examples/common:dot_product_8f.mlir"/>
    </rule>
</query>"""
      elif (
          "kind(heir_opt, //tests/Examples/openfhe/ckks/dot_product_8f:*)"  # fmt: skip
          in query_str
      ):
        return "//tests/Examples/openfhe/ckks/dot_product_8f:dot_product_8f_test_heir_opt"
      return ""

    mock_run_blaze_query.side_effect = side_effect

    f = io.StringIO()
    with redirect_stdout(f):
      e2e_to_bazel_lib.e2e_to_bazel(
          "//tests/Examples/openfhe/ckks/dot_product_8f:dot_product_8f_test"
      )
    output = f.getvalue().strip()

    expected_command = (
        "bazel run --noallow_analysis_cache_discard //tools:heir-opt --"
        " '--annotate-module=backend=openfhe scheme=ckks'"
        " --mlir-to-ckks=ciphertext-degree=1024 --scheme-to-openfhe"
        " /workspace/tests/Examples/common/dot_product_8f.mlir"
    )
    self.assertIn(expected_command, output)

  @patch("scripts.e2e_to_bazel_lib.run_blaze_query")
  @absltest.mock.patch.dict(
      "os.environ", {"BUILD_WORKSPACE_DIRECTORY": "/workspace"}
  )
  def test_e2e_to_bazel_with_file(self, mock_run_blaze_query):
    """Tests e2e_to_bazel with a file argument."""

    def side_effect(query_str, options=None):
      if options and "--output=xml" in options:
        return """<?xml version="1.1" encoding="UTF-8" standalone="no"?>
<query version="2">
    <rule class="heir_opt" name="//tests/Examples/openfhe/ckks/dot_product_8f:dot_product_8f_test_heir_opt">
        <list name="pass_flags">
            <string value="--annotate-module=backend=openfhe scheme=ckks"/>
            <string value="--mlir-to-ckks=ciphertext-degree=1024"/>
            <string value="--scheme-to-openfhe"/>
        </list>
        <label name="src" value="//tests/Examples/common:dot_product_8f.mlir"/>
    </rule>
</query>"""
      elif "rdeps(" in query_str:
        return "//tests/Examples/openfhe/ckks/dot_product_8f:dot_product_8f_test_heir_opt"
      return ""

    mock_run_blaze_query.side_effect = side_effect

    f = io.StringIO()
    with redirect_stdout(f):
      e2e_to_bazel_lib.e2e_to_bazel("tests/Examples/common/dot_product_8f.mlir")  # fmt: skip
    output = f.getvalue().strip()

    expected_command = (
        "bazel run --noallow_analysis_cache_discard //tools:heir-opt --"
        " '--annotate-module=backend=openfhe scheme=ckks'"
        " --mlir-to-ckks=ciphertext-degree=1024 --scheme-to-openfhe"
        " /workspace/tests/Examples/common/dot_product_8f.mlir"
    )
    self.assertIn(expected_command, output)

  def test_path_to_label_with_workspace_dir(self):
    """Tests for path_to_label with BUILD_WORKSPACE_DIRECTORY."""

    def side_effect(path):
      if path == "tests/foo.mlir":
        return False
      if path == "/workspace/tests/foo.mlir":
        return True
      if path == "/workspace/tests/BUILD":
        return True
      return False

    with patch("os.path.isfile", side_effect=side_effect):
      with patch("os.path.exists", side_effect=side_effect):
        with patch.dict(
            "os.environ", {"BUILD_WORKSPACE_DIRECTORY": "/workspace"}
        ):
          self.assertEqual(
              e2e_to_bazel_lib.path_to_label("tests/foo.mlir"),
              "//tests:foo.mlir",
          )

  def test_path_to_label_with_workspace_dir_and_prefix(self):
    """Tests for path_to_label with BUILD_WORKSPACE_DIRECTORY and prefix removal."""

    def side_effect(path):
      if path == "third_party/heir/tests/foo.mlir":
        return False
      if path == "/workspace/tests/foo.mlir":
        return True
      if path == "/workspace/tests/BUILD":
        return True
      return False

    with patch("os.path.isfile", side_effect=side_effect):
      with patch("os.path.exists", side_effect=side_effect):
        with patch.dict(
            "os.environ", {"BUILD_WORKSPACE_DIRECTORY": "/workspace"}
        ):
          self.assertEqual(
              e2e_to_bazel_lib.path_to_label("third_party/heir/tests/foo.mlir"),
              "//tests:foo.mlir",
          )


if __name__ == "__main__":
  absltest.main()
