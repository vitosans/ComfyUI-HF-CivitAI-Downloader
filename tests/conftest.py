"""Make the pack importable as `ComfyUI_HuggingFace_Downloader` whatever its folder is called.

The tests (shared with upstream) import it by the upstream folder name; this fork's folder is
`ComfyUI-HF-CivitAI-Downloader`, which is not a valid module name. Only a package *path* is
registered here; nothing is imported until a test does it (its own sys.modules stubs come first).
"""
import importlib.machinery
import importlib.util
import os
import sys

PACKAGE = "ComfyUI_HuggingFace_Downloader"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _AliasFinder:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name != PACKAGE:
            return None
        return importlib.util.spec_from_file_location(
            PACKAGE,
            os.path.join(REPO_DIR, "__init__.py"),
            submodule_search_locations=[REPO_DIR],
        )


if os.path.basename(REPO_DIR) != PACKAGE:
    sys.meta_path.insert(0, _AliasFinder())
