import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

for mod_name in ["comfy_api", "comfy_api.latest", "folder_paths", "server", "aiohttp", "aiohttp.web"]:
    sys.modules.setdefault(mod_name, MagicMock())

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(repo_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import ComfyUI_HuggingFace_Downloader.downloader as dl


class TestCivitaiSettings(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CIVITAI_API_TOKEN", None)

    def _settings(self, value):
        p = patch.object(dl, "_load_comfy_settings_dict", return_value=value)
        p.start()
        self.addCleanup(p.stop)

    def test_token_from_settings(self):
        self._settings({"downloader.civitai_token": " abc "})
        self.assertEqual(dl.get_civitai_token(), "abc")

    def test_token_from_env(self):
        self._settings({})
        os.environ["CIVITAI_API_TOKEN"] = " envkey "
        with patch.object(dl, "_lora_manager_settings_path", return_value="/nonexistent/x.json"):
            self.assertEqual(dl.get_civitai_token(), "envkey")

    def test_token_falls_back_to_lora_manager(self):
        self._settings({})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            with open(path, "w") as f:
                json.dump({"civitai_api_key": "lm"}, f)
            with patch.object(dl, "_lora_manager_settings_path", return_value=path):
                self.assertEqual(dl.get_civitai_token(), "lm")

    def test_bad_lora_manager_file_is_skipped(self):
        self._settings({})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "settings.json")
            with open(path, "w") as f:
                f.write("{not json")
            with patch.object(dl, "_lora_manager_settings_path", return_value=path):
                self.assertIsNone(dl.get_civitai_token())

    def test_token_none_when_absent(self):
        self._settings({})
        with patch.object(dl, "_lora_manager_settings_path", return_value="/nonexistent/x.json"):
            self.assertIsNone(dl.get_civitai_token())

    def test_search_enabled_default_true(self):
        self._settings({})
        self.assertTrue(dl.civitai_search_enabled())

    def test_search_disabled(self):
        self._settings({"downloader.civitai_search_enabled": False})
        self.assertFalse(dl.civitai_search_enabled())

    def test_constants(self):
        self.assertEqual(dl.CIVITAI_API, "https://civitai.com")
        self.assertEqual(dl.ARCHIVE_API, "https://civitaiarchive.com")
        self.assertEqual(dl.CIVITAI_HOSTS, {"civitai.com", "civitai.red", "civitai.green"})


if __name__ == "__main__":
    unittest.main()
