import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

for mod_name in ["comfy_api", "comfy_api.latest", "folder_paths", "server", "aiohttp", "aiohttp.web"]:
    sys.modules.setdefault(mod_name, MagicMock())

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(repo_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import ComfyUI_HuggingFace_Downloader.backup as backup

SETTINGS = {"downloader.hf_token": "hf_x", "downloader.civitai_token": "cv_x", "other": 1}


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestBackupSecrets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name

    def test_secret_keys(self):
        self.assertEqual(set(backup.SECRET_SETTING_KEYS),
                         {"downloader.hf_token", "downloader.civitai_token"})

    def test_single_settings_copy_strips_all_secrets(self):
        src = os.path.join(self.root, "src", "comfy.settings.json")
        _write(src, SETTINGS)
        out_dir = os.path.join(self.root, "out")
        os.makedirs(out_dir)
        self.assertEqual(_read(backup._copy_settings_without_token(src, out_dir)), {"other": 1})

    def test_folder_copy_strips_all_secrets(self):
        src = os.path.join(self.root, "user")
        _write(os.path.join(src, "default", "comfy.settings.json"), SETTINGS)
        out_dir = os.path.join(self.root, "out")
        os.makedirs(out_dir)
        dst = backup._copy_and_strip_token(src, out_dir)
        self.assertEqual(_read(os.path.join(dst, "default", "comfy.settings.json")), {"other": 1})

    def test_merge_keeps_local_secrets(self):
        merged = backup._merge_restored_settings(
            {"other": 2, "downloader.civitai_token": "from_backup"}, SETTINGS, "fallback")
        self.assertEqual(merged, {"other": 2, "downloader.hf_token": "hf_x", "downloader.civitai_token": "cv_x"})

    def test_merge_without_local_secrets(self):
        new = {"other": 2, "downloader.civitai_token": "from_backup", "downloader.hf_token": "old"}
        merged = backup._merge_restored_settings(new, {}, "fallback")
        self.assertEqual(merged, {"other": 2, "downloader.hf_token": "fallback"})
        self.assertIn("downloader.civitai_token", new)  # input not mutated

    def test_restore_selected_file_preserves_civitai_token(self):
        target = os.path.join(self.root, "comfy")
        dst = os.path.join(target, "user", "default", "comfy.settings.json")
        _write(dst, SETTINGS)
        src = os.path.join(self.root, "dl", "comfy.settings.json")
        _write(src, {"other": 3})
        backup._copy_repo_file_to_target(src, "ComfyUI/user/default/comfy.settings.json", target, "tok")
        self.assertEqual(_read(dst), {"other": 3, "downloader.hf_token": "hf_x", "downloader.civitai_token": "cv_x"})


if __name__ == "__main__":
    unittest.main()
