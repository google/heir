import importlib.util
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from heir.backends.openfhe import config


class ConfigTest(unittest.TestCase):

  def test_from_os_env_only_debug(self):
    with mock.patch.dict(os.environ, {"OPENFHE_DEBUG": "1"}, clear=True):
      self.assertIsNone(config.from_os_env())

  def test_from_os_env_valid(self):
    env = {
        "OPENFHE_INCLUDE_DIR": "/fake/include",
        "OPENFHE_LIB_DIR": "/fake/lib",
    }
    with mock.patch.dict(os.environ, env, clear=True):
      res = config.from_os_env()
      self.assertIsNotNone(res)
      self.assertEqual(res.include_dirs, ["/fake/include"])
      self.assertEqual(res.lib_dir, "/fake/lib")

  def test_from_os_env_appends_parent_of_openfhe_dir(self):
    env = {
        "OPENFHE_INCLUDE_DIR": "/fake/include/openfhe",
        "OPENFHE_LIB_DIR": "/fake/lib",
    }
    with mock.patch.dict(os.environ, env, clear=True):
      res = config.from_os_env()
      self.assertIsNotNone(res)
      self.assertEqual(
          res.include_dirs, ["/fake/include/openfhe", "/fake/include"]
      )

  def test_from_os_env_missing_lib(self):
    env = {
        "OPENFHE_INCLUDE_DIR": "/fake/include",
    }
    with mock.patch.dict(os.environ, env, clear=True):
      with self.assertRaises(ValueError):
        config.from_os_env()


class ResolveLinkLibsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp_dir.cleanup)
    self.lib_path = pathlib.Path(self.temp_dir.name)

  def test_resolve_split_libs_lowercase(self):
    core = self.lib_path / "libopenfhecore.so"
    core.touch()
    pke = self.lib_path / "libopenfhepke.so"
    pke.touch()
    binfhe = self.lib_path / "libopenfhebinfhe.so"
    binfhe.touch()

    libs = config._resolve_link_libs(self.lib_path)
    self.assertEqual(
        libs, [str(core.resolve()), str(pke.resolve()), str(binfhe.resolve())]
    )

  def test_resolve_split_libs_uppercase(self):
    core = self.lib_path / "libOPENFHEcore.so.1"
    core.touch()
    pke = self.lib_path / "libOPENFHEpke.so.1"
    pke.touch()
    binfhe = self.lib_path / "libOPENFHEbinfhe.so.1"
    binfhe.touch()

    libs = config._resolve_link_libs(self.lib_path)
    self.assertEqual(
        libs, [str(core.resolve()), str(pke.resolve()), str(binfhe.resolve())]
    )

  def test_resolve_monolithic_lib(self):
    mono = self.lib_path / "libopenfhe.so"
    mono.touch()

    core = self.lib_path / "libopenfhecore.so"
    core.touch()

    libs = config._resolve_link_libs(self.lib_path)
    self.assertEqual(libs, [str(mono.resolve())])

  def test_resolve_fallback(self):
    libs = config._resolve_link_libs(self.lib_path)
    self.assertEqual(libs, ["openfhe"])


class ResolveConfigTest(unittest.TestCase):

  @mock.patch.object(config, "from_os_env")
  @mock.patch.object(config, "get_default_installed_config")
  @mock.patch.object(config, "development_openfhe_config")
  def test_cascade_os_env(self, mock_dev, mock_default, mock_os_env):
    mock_os_env.return_value = "os_env"

    res = config.resolve_config()
    self.assertEqual(res, "os_env")
    mock_os_env.assert_called_once()
    mock_default.assert_not_called()
    mock_dev.assert_not_called()

  @mock.patch.object(config, "from_os_env")
  @mock.patch.object(config, "get_default_installed_config")
  @mock.patch.object(config, "development_openfhe_config")
  def test_cascade_default_installed(self, mock_dev, mock_default, mock_os_env):
    mock_os_env.return_value = None
    mock_default.return_value = "default_installed"

    res = config.resolve_config()
    self.assertEqual(res, "default_installed")
    mock_os_env.assert_called_once()
    mock_default.assert_called_once()
    mock_dev.assert_not_called()

  @mock.patch.object(config, "from_os_env")
  @mock.patch.object(config, "get_default_installed_config")
  @mock.patch.object(config, "development_openfhe_config")
  def test_cascade_development(self, mock_dev, mock_default, mock_os_env):
    mock_os_env.return_value = None
    mock_default.return_value = None
    mock_dev.return_value = "development"

    res = config.resolve_config()
    self.assertEqual(res, "development")
    mock_os_env.assert_called_once()
    mock_default.assert_called_once()
    mock_dev.assert_called_once()


if __name__ == "__main__":
  unittest.main()
