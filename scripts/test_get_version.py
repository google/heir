"""Tests for get_version.py."""

from unittest.mock import MagicMock, patch
from absl.testing import absltest
from scripts import get_version

calculate_version = get_version.calculate_version
get_next_dev_version = get_version.get_next_dev_version
get_pypi_versions = get_version.get_pypi_versions


class GetVersionTest(absltest.TestCase):

  @patch("requests.get")
  def test_get_pypi_versions_success(self, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "releases": {
            "0.1.0": [],
            "0.1.1": [],
        }
    }
    mock_get.return_value = mock_response

    versions = get_pypi_versions("heir_py")
    self.assertIn("0.1.0", versions)
    self.assertIn("0.1.1", versions)
    self.assertEqual(len(versions), 2)

  @patch("requests.get")
  def test_get_pypi_versions_failure(self, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    versions = get_pypi_versions("heir_py")
    self.assertEqual(versions, [])

  @patch("scripts.get_version.get_pypi_versions")
  @patch("datetime.datetime")
  def test_get_next_dev_version_none_today(self, mock_datetime, mock_get_pypi):
    mock_datetime.now.return_value.strftime.return_value = "2026.04.01"
    mock_get_pypi.return_value = ["2026.03.31.dev0"]

    next_version = get_next_dev_version("heir_py")
    self.assertEqual(next_version, "2026.04.01.dev0")

  @patch("scripts.get_version.get_pypi_versions")
  @patch("datetime.datetime")
  def test_get_next_dev_version_exists_today(
      self, mock_datetime, mock_get_pypi
  ):
    mock_datetime.now.return_value.strftime.return_value = "2026.04.01"
    mock_get_pypi.return_value = ["2026.04.01.dev0", "2026.04.01.dev1"]

    next_version = get_next_dev_version("heir_py")
    self.assertEqual(next_version, "2026.04.01.dev2")

  @patch("scripts.get_version.get_pypi_versions")
  @patch("datetime.datetime")
  def test_get_next_dev_version_multiple_years(
      self, mock_datetime, mock_get_pypi
  ):
    mock_datetime.now.return_value.strftime.return_value = "2026.04.01"
    mock_get_pypi.return_value = ["2025.04.01.dev0", "2026.04.01.dev0"]

    next_version = get_next_dev_version("heir_py")
    self.assertEqual(next_version, "2026.04.01.dev1")

  def test_calculate_version_release(self):
    version, should_publish = calculate_version(
        "release", "refs/tags/v0.1.0", "v0.1.0", "heir_py"
    )
    self.assertEqual(version, "0.1.0")
    self.assertEqual(should_publish, "true")

  def test_calculate_version_workflow_dispatch_tag(self):
    version, should_publish = calculate_version(
        "workflow_dispatch", "refs/heads/main", "v0.1.0", "heir_py"
    )
    self.assertEqual(version, "0.1.0")
    self.assertEqual(should_publish, "true")

  @patch("scripts.get_version.get_next_dev_version")
  def test_calculate_version_workflow_dispatch_main(self, mock_get_next_dev):
    mock_get_next_dev.return_value = "2026.04.01.dev0"
    version, should_publish = calculate_version(
        "workflow_dispatch", "refs/heads/main", None, "heir_py"
    )
    self.assertEqual(version, "2026.04.01.dev0")
    self.assertEqual(should_publish, "true")

  def test_calculate_version_pr(self):
    version, should_publish = calculate_version(
        "pull_request", "refs/pull/123/merge", None, "heir_py"
    )
    self.assertEqual(version, "0.0.0")
    self.assertEqual(should_publish, "false")


if __name__ == "__main__":
  absltest.main()
