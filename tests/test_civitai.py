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

import ComfyUI_HuggingFace_Downloader.civitai_discovery as cd
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




class TestSearchQueries(unittest.TestCase):
    def test_camel_case_and_digits_split_into_words(self):
        qs = cd.search_queries("moodyKrea2MixUncensoredV8SEGoes_v80_int8.safetensors")
        self.assertEqual(qs[0], "moody krea 2 mix uncensored v8se goes")
        self.assertIn("moody krea 2 mix", qs)

    def test_basename_with_backslashes(self):
        self.assertEqual(cd.search_queries("Krea2\\sub/foo_bar_v1.safetensors")[0], "foo bar")

    def test_precision_and_format_tokens_dropped(self):
        self.assertEqual(cd.search_queries("cyberrealistic_v90_fp8_e4m3fn.gguf")[0], "cyberrealistic")

    def test_too_short_yields_nothing(self):
        self.assertEqual(cd.search_queries("x.safetensors"), [])

    def test_max_three_deduped_queries(self):
        qs = cd.search_queries("moodyKrea2MixUncensoredV8SEGoes.safetensors")
        self.assertLessEqual(len(qs), 3)
        self.assertEqual(len(qs), len(set(qs)))


FIXTURES = os.path.join(current_dir, "fixtures", "civitai")
HASH = "7b9c56858a518f9fbed261aa8c535d3bb409fc8ecf441dad780902703c1ed454"
RENAMED = "moodyKrea2MixUncensoredV8SEGoes_v80_int8.safetensors"


def _fx(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


class FakeApi:
    """Callable standing in for cd._get_json: url-substring -> payload (or None)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, auth=False):
        self.calls.append(url)
        for needle, payload in self.routes:
            if needle in url:
                return payload
        return None


def _routes(search=True, by_hash=True, archive_search=True, archive_sha=True):
    return [
        ("/api/v1/models?", _fx("models_search_moody.json") if search else {"items": []}),
        ("by-hash", _fx("model_version_by_hash.json") if by_hash else None),
        ("/api/search", _fx("archive_search.json") if archive_search else {"results": []}),
        ("/api/sha256/", _fx("archive_sha256.json") if archive_sha else None),
    ]


class TestLookup(unittest.TestCase):
    def _run(self, filename, routes, budget=None, status=None):
        api = FakeApi(routes)
        with patch.object(cd, "_get_json", api):
            result = cd.lookup(filename, budget or cd.Budget(), status)
        return result, api

    def test_exact_filename_in_family(self):
        result, _ = self._run("moodyKrea2MixV9IsHere_v80_int8.safetensors", _routes())
        exact = result["exact"]
        self.assertEqual(exact["match"], "exact")
        self.assertEqual(exact["site"], "civitai")
        self.assertTrue(exact["url"].startswith("https://civitai.com/api/download/models/3327244"))
        self.assertEqual(exact["sha256"], HASH)
        self.assertEqual(exact["base_model"], "Krea 2")
        self.assertEqual(exact["version_name"], "V8.0")
        self.assertIsInstance(exact["size_bytes"], int)

    def test_renamed_file_found_by_hash(self):
        result, api = self._run(RENAMED, _routes())
        exact = result["exact"]
        self.assertEqual(exact["match"], "hash")
        self.assertEqual(exact["sha256"], HASH)
        self.assertEqual(exact["filename"], "moodyKrea2MixV9IsHere_v80_int8.safetensors")
        self.assertFalse(any("by-hash" in u for u in api.calls))

    def test_hash_falls_back_to_by_hash_when_search_misses(self):
        result, api = self._run(RENAMED, _routes(search=False))
        exact = result["exact"]
        self.assertEqual(exact["match"], "hash")
        self.assertEqual(exact["site"], "civitai")
        self.assertEqual(exact["sha256"], HASH)
        self.assertTrue(any("by-hash/" + HASH in u for u in api.calls))

    def test_archive_mirror_when_civitai_lost_it(self):
        result, _ = self._run(RENAMED, _routes(search=False, by_hash=False))
        exact = result["exact"]
        self.assertEqual(exact["site"], "civitai_archive")
        self.assertEqual(exact["match"], "hash")
        self.assertEqual(exact["sha256"], HASH)
        mirror = next(
            m for m in _fx("archive_sha256.json")["files"]
            if m["deletedAt"] is None and not m["is_gated"] and not m["is_paid"]
        )
        self.assertEqual(exact["url"], mirror["url"])

    def test_family_same_base_only(self):
        search = _fx("models_search_moody.json")
        other = json.loads(json.dumps(search["items"][1]))
        other["id"] = 99
        for v in other["modelVersions"]:
            v["baseModel"] = "Flux.1 D"
        search["items"].append(other)
        routes = [("/api/v1/models?", search), ("/api/search", _fx("archive_search.json"))]
        result, _ = self._run("moodyKrea2MixV9IsHere_vUnknown_int8.safetensors", routes)
        self.assertIsNone(result["exact"])
        fam = result["family"]
        self.assertTrue(0 < len(fam) <= 8)
        self.assertTrue(all(c["base_model"] == "Krea 2" for c in fam))
        self.assertTrue(all(c["match"] == "family" for c in fam))
        self.assertEqual(fam[0]["model_name"], search["items"][0]["name"])
        self.assertNotIn("Flux.1 D", [c["base_model"] for c in fam])

    def test_family_unknown_base_same_model_only(self):
        search = _fx("models_search_moody.json")
        search["items"] = search["items"][:2]
        routes = [("/api/v1/models?", search), ("/api/search", {"results": []})]
        result, _ = self._run("moodyKrea2MixV9IsHere_vUnknown_int8.safetensors", routes)
        self.assertIsNone(result["exact"])
        self.assertTrue(result["family"])
        self.assertEqual({c["model_name"] for c in result["family"]}, {search["items"][0]["name"]})

    def test_only_model_extensions_are_candidates(self):
        result, _ = self._run("moodyKrea2MixV9IsHere_vUnknown_int8.safetensors", _routes(archive_search=False))
        exts = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft")
        self.assertTrue(all(c["filename"].lower().endswith(exts) for c in result["family"]))

    def test_unauthorized_degrades(self):
        routes = [
            ("civitai.com", None),
            ("/api/search", _fx("archive_search.json")),
            ("/api/sha256/", _fx("archive_sha256.json")),
        ]
        result, _ = self._run(RENAMED, routes)
        self.assertEqual(result["exact"]["site"], "civitai_archive")

    def test_malformed_payload_skipped(self):
        search = {"items": [
            {"id": 1, "name": "bad", "modelVersions": [{"files": None}]},
            {"id": 2, "name": "worse", "modelVersions": None},
            "junk",
            _fx("models_search_moody.json")["items"][0],
        ]}
        routes = [("/api/v1/models?", search), ("/api/search", {"results": "x"})]
        result, _ = self._run("moodyKrea2MixV9IsHere_v80_int8.safetensors", routes)
        self.assertEqual(result["exact"]["match"], "exact")

    def test_no_results_returns_none(self):
        result, _ = self._run("moodyKrea2MixV9IsHere_v80_int8.safetensors", [])
        self.assertIsNone(result)

    def test_budget_stops_requests(self):
        _, api = self._run(RENAMED, _routes(), budget=cd.Budget(max_requests=1))
        self.assertEqual(len(api.calls), 1)

    def test_rate_limit_stops_lookup(self):
        calls = []

        def boom(url, auth=False):
            calls.append(url)
            raise cd.RateLimited()

        with patch.object(cd, "_get_json", boom):
            result = cd.lookup(RENAMED, cd.Budget())
        self.assertIsNone(result)
        self.assertEqual(len(calls), 1)

    def test_status_sources(self):
        seen = []
        self._run(RENAMED, _routes(search=False), status=seen.append)
        self.assertEqual(seen[0], "civitai_search")
        self.assertIn("civitai_archive", seen)
        self.assertIn("civitai_hash", seen)


class TestLookupRobustness(unittest.TestCase):
    _run = TestLookup._run

    def test_rate_limit_exhausts_shared_budget(self):
        budget = cd.Budget()
        calls = []

        def boom(url, auth=False):
            calls.append(url)
            raise cd.RateLimited()

        with patch.object(cd, "_get_json", boom):
            cd.lookup(RENAMED, budget)
            n = len(calls)
            cd.lookup(RENAMED, budget)
        self.assertEqual(len(calls), n)

    def test_archive_base_ignored_for_nonmatching_name(self):
        archive = {"results": [{"name": "other.safetensors", "url": "/sha256/ab", "base_model": "Flux.1 D"}]}
        search = _fx("models_search_moody.json")
        result, _ = self._run("moodyKrea2MixV9IsHere_vUnknown_int8.safetensors",
                              [("/api/v1/models?", search), ("/api/search", archive)])
        self.assertTrue(result["family"])
        self.assertNotIn("Flux.1 D", [c["base_model"] for c in result["family"]])
        self.assertTrue(all(c["base_model"] == "Krea 2" for c in result["family"]))

    def test_top_model_uniform_base_used_for_family(self):
        search = _fx("models_search_moody.json")
        search["items"][0]["modelVersions"] = search["items"][0]["modelVersions"][:1]
        result, _ = self._run("moodyKrea2MixV9IsHere_vUnknown_int8.safetensors",
                              [("/api/v1/models?", search), ("/api/search", {"results": []})])
        self.assertGreater(len({c["model_name"] for c in result["family"]}), 1)

    def test_top_model_mixed_base_gives_same_model_family_only(self):
        search = _fx("models_search_moody.json")
        search["items"][0]["modelVersions"][0]["baseModel"] = "Flux.1 D"
        result, _ = self._run("moodyKrea2MixV9IsHere_vUnknown_int8.safetensors",
                              [("/api/v1/models?", search), ("/api/search", {"results": []})])
        self.assertEqual({c["model_name"] for c in result["family"]}, {search["items"][0]["name"]})

    def test_partial_family_returned_when_budget_runs_out(self):
        result, _ = self._run(RENAMED, _routes(search=True), budget=cd.Budget(max_requests=1))
        self.assertIsNone(result["exact"])
        self.assertTrue(result["family"])

    def test_exact_file_not_in_family_without_sha(self):
        search = _fx("models_search_moody.json")
        name = "moodyKrea2MixV9IsHere_v80_int8.safetensors"
        for it in search["items"]:
            for v in it["modelVersions"]:
                for f in v["files"]:
                    if f["name"] == name:
                        f["hashes"] = {}
        result, _ = self._run(name, [("/api/v1/models?", search)])
        self.assertEqual(result["exact"]["filename"], name)
        self.assertNotIn(name, [c["filename"] for c in result["family"]])


class TestGetJson(unittest.TestCase):
    class _Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _captured(self, url, auth, token="secret"):
        seen = {}

        def fake_open(req, timeout=None):
            seen["req"] = req
            return self._Resp(b'{"ok": 1}')

        with patch.object(cd, "get_civitai_token", return_value=token), \
                patch.object(cd.urllib.request, "urlopen", fake_open):
            out = cd._get_json(url, auth)
        return out, seen["req"]

    def test_token_sent_only_to_civitai_hosts_when_auth(self):
        out, req = self._captured("https://civitai.com/api/v1/models?x=1", True)
        self.assertEqual(out, {"ok": 1})
        self.assertEqual(req.unredirected_hdrs.get("Authorization"), "Bearer secret")

    def test_token_never_sent_to_archive(self):
        _, req = self._captured("https://civitaiarchive.com/api/search?q=a", True)
        self.assertNotIn("Authorization", req.unredirected_hdrs)
        self.assertNotIn("Authorization", req.headers)

    def test_no_token_without_auth_flag(self):
        _, req = self._captured("https://civitai.com/api/v1/models", False)
        self.assertNotIn("Authorization", req.unredirected_hdrs)

    def test_429_raises_and_other_errors_return_none(self):
        def raiser(code):
            def f(req, timeout=None):
                raise cd.urllib.error.HTTPError(req.full_url, code, "x", {}, None)
            return f

        with patch.object(cd.urllib.request, "urlopen", raiser(429)):
            with self.assertRaises(cd.RateLimited):
                cd._get_json("https://civitai.com/x", False)
        with patch.object(cd.urllib.request, "urlopen", raiser(401)):
            self.assertIsNone(cd._get_json("https://civitai.com/x", False))

    def test_incomplete_read_returns_none(self):
        def f(req, timeout=None):
            raise cd.http.client.IncompleteRead(b"")
        with patch.object(cd.urllib.request, "urlopen", f):
            self.assertIsNone(cd._get_json("https://civitai.com/x", False))

    def test_invalid_json_returns_none(self):
        with patch.object(cd.urllib.request, "urlopen", lambda r, timeout=None: self._Resp(b"<html>")):
            self.assertIsNone(cd._get_json("https://civitai.com/x", False))


class TestBudget(unittest.TestCase):
    def test_spend_until_exhausted(self):
        b = cd.Budget(max_requests=2, deadline_s=60)
        self.assertEqual([b.spend(), b.spend(), b.spend()], [True, True, False])

    def test_deadline(self):
        self.assertFalse(cd.Budget(max_requests=5, deadline_s=0).spend())


if __name__ == "__main__":
    unittest.main()
