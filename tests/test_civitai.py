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
    def _run(self, filename, routes, budget=None, status=None, folder=None):
        api = FakeApi(routes)
        with patch.object(cd, "_get_json", api):
            result = cd.lookup(filename, budget or cd.Budget(), status, folder=folder)
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


def _cand(**kw):
    base = {
        "filename": "moody_v9.safetensors", "url": "https://civitai.com/api/download/models/1",
        "sha256": "a" * 64, "size_bytes": int(13.5 * 1024 ** 3), "base_model": "Krea 2",
        "version_name": "V8.0", "model_name": "Moody Krea 2 Mix - V9 is here", "nsfw": True,
        "site": "civitai", "match": "exact", "civitai_type": "Model",
    }
    base.update(kw)
    return base


class TestSearchMissing(unittest.TestCase):
    def setUp(self):
        p = patch.object(dl, "civitai_search_enabled", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    def _run(self, rows, result, **kw):
        kw.setdefault("skip_all", False)
        kw.setdefault("skip_filenames", set())
        kw.setdefault("request_id", None)
        with patch.object(cd, "lookup", return_value=result) as lk:
            cd.search_missing(rows, **kw)
        return lk

    def test_unresolved_row_gets_civitai_url(self):
        rows = [{"filename": "moody_v9.safetensors"}]
        self._run(rows, {"exact": _cand(), "family": []})
        r = rows[0]
        self.assertEqual(r["filename"], "moody_v9.safetensors")
        self.assertEqual(r["url"], "https://civitai.com/api/download/models/1")
        self.assertEqual(r["sha256"], "a" * 64)
        self.assertEqual(r["source"], "civitai")

    def test_archive_exact_sets_archive_source(self):
        rows = [{"filename": "x.safetensors"}]
        self._run(rows, {"exact": _cand(site="civitai_archive"), "family": []})
        self.assertEqual(rows[0]["source"], "civitai_archive")

    def _hf_row(self):
        return {"filename": "moody_v9.safetensors", "url": "https://huggingface.co/a/b/resolve/main/m.safetensors",
                "hf_repo": "a/b", "hf_path": "m.safetensors", "source": "huggingface_search",
                "alternatives": [{"filename": "old", "url": "u", "source": "popular_models"}]}

    def test_hf_row_keeps_hf_primary_and_gets_alternative(self):
        for sha, tail in (("a" * 64, "same file"), ("b" * 64, "different file")):
            rows = [self._hf_row()]
            self._run(rows, {"exact": _cand(), "family": []}, hf_sha256=lambda repo, path, s=sha: s)
            r = rows[0]
            self.assertTrue(r["url"].startswith("https://huggingface.co/"))
            self.assertEqual(r["source"], "huggingface_search")
            self.assertEqual(len(r["alternatives"]), 2)
            alt = r["alternatives"][1]
            self.assertEqual(alt["source"], "civitai")
            self.assertTrue(alt["label"].endswith(tail), alt["label"])

    def test_label_format(self):
        rows = [self._hf_row()]
        self._run(rows, {"exact": _cand(), "family": []}, hf_sha256=lambda r, p: None)
        self.assertEqual(rows[0]["alternatives"][1]["label"],
                         "CivitAI \u00b7 Moody Krea 2 Mix - V9 is here V8.0 \u00b7 moody_v9.safetensors"
                         " \u00b7 13.5 GB \u00b7 Krea 2 \u00b7 NSFW")

    def test_archive_label_and_unknown_parts_omitted(self):
        rows = [{"filename": "q.safetensors"}]
        fam = _cand(site="civitai_archive", size_bytes=None, base_model=None, nsfw=False,
                    version_name="", model_name="M", match="family")
        self._run(rows, {"exact": None, "family": [fam]})
        self.assertEqual(rows[0]["alternatives"][0]["label"], "CivitAI Archive \u00b7 M \u00b7 moody_v9.safetensors")

    def test_family_alternatives_keep_real_filename(self):
        rows = [{"filename": "moody_v9.safetensors", "suggested_folder": "diffusion_models"}]
        fams = [_cand(filename=f"real_{i}.safetensors", match="family", url=f"https://civitai.com/d/{i}")
                for i in range(2)]
        self._run(rows, {"exact": None, "family": fams})
        r = rows[0]
        self.assertNotIn("url", r)
        self.assertEqual([a["filename"] for a in r["alternatives"]], ["real_0.safetensors", "real_1.safetensors"])
        self.assertEqual(r["alternatives"][0]["suggested_folder"], "diffusion_models")
        self.assertEqual(r["alternatives"][0]["url"], "https://civitai.com/d/0")

    def test_non_hf_url_untouched(self):
        rows = [{"filename": "a.safetensors", "url": "https://example.com/a.safetensors"}]
        lk = self._run(rows, {"exact": _cand(), "family": [_cand(match="family")]})
        self.assertEqual(rows, [{"filename": "a.safetensors", "url": "https://example.com/a.safetensors"}])
        lk.assert_not_called()

    def test_skip_all_makes_no_lookup(self):
        lk = self._run([{"filename": "a.safetensors"}], None, skip_all=True)
        lk.assert_not_called()

    def test_skip_filename_respected(self):
        lk = self._run([{"filename": "A.safetensors"}], None, skip_filenames={"a.safetensors"})
        lk.assert_not_called()

    def test_disabled_setting_makes_no_lookup(self):
        with patch.object(dl, "civitai_search_enabled", return_value=False):
            lk = self._run([{"filename": "a.safetensors"}], None)
        lk.assert_not_called()

    def test_no_result_leaves_row(self):
        rows = [{"filename": "a.safetensors"}]
        self._run(rows, None)
        self.assertEqual(rows, [{"filename": "a.safetensors"}])

    def test_rate_limit_stops_remaining_lookups(self):
        budget_seen = []

        def fake(name, budget, status_cb=None, **kw):
            budget_seen.append(budget)
            budget.exhaust()
            return None

        rows = [{"filename": "a.safetensors"}, {"filename": "b.safetensors"}]
        with patch.object(cd, "lookup", fake):
            cd.search_missing(rows, skip_all=False, skip_filenames=set(), request_id=None)
        self.assertEqual(len(budget_seen), 1)

    def test_status_callback_receives_dict(self):
        events = []

        def fake(name, budget, status_cb=None, **kw):
            status_cb("civitai_search")
            return None

        with patch.object(cd, "lookup", fake):
            cd.search_missing([{"filename": "a.safetensors"}], skip_all=False, skip_filenames=set(),
                              request_id=None, status_cb=events.append)
        self.assertEqual(events[0]["source"], "civitai_search")
        self.assertEqual(events[0]["filename"], "a.safetensors")

    def test_nothing_missing_no_lookup(self):
        import ComfyUI_HuggingFace_Downloader.model_discovery as md
        with patch.object(md, "check_model_files", return_value=([], [], [])), \
                patch.object(cd, "search_missing") as sm:
            md.process_workflow_for_missing_models({"prefilled_missing_models": []})
        sm.assert_not_called()


class _FakeResponse:
    def __init__(self, body, url):
        import io
        self._io = io.BytesIO(body)
        self._url = url
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n=-1):
        return self._io.read(n)

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestDirectDownload(unittest.TestCase):
    def setUp(self):
        import tempfile as _tf
        self.tmp = _tf.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.requests = []
        for p in (
            patch.object(dl, "resolve_target_dir", return_value=self.tmp.name),
            patch.object(dl, "is_path_within_allowed_roots", return_value=True),
            patch.object(dl, "get_civitai_token", return_value="k"),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _run(self, url, body=b"abc", **kw):
        def fake_urlopen(req, timeout=None):
            self.requests.append(req)
            return _FakeResponse(body, url)
        with patch.object(dl.urllib.request, "urlopen", side_effect=fake_urlopen):
            return dl.run_download_url(url, "loras", sync=True, target_filename="m.safetensors", **kw)

    def test_civitai_request_has_unredirected_auth(self):
        self._run("https://civitai.com/api/download/models/1")
        req = self.requests[0]
        self.assertEqual(req.unredirected_hdrs.get("Authorization"), "Bearer k")
        self.assertNotIn("Authorization", req.headers)

    def test_www_civitai_host_gets_auth(self):
        self._run("https://www.civitai.red/api/download/models/1")
        self.assertEqual(self.requests[0].unredirected_hdrs.get("Authorization"), "Bearer k")

    def test_non_civitai_has_no_auth(self):
        for url in ("https://civitaiarchive.com/api/download/models/1", "https://cdn.example/x.safetensors"):
            self.requests.clear()
            self._run(url, overwrite=True)
            req = self.requests[0]
            self.assertNotIn("Authorization", req.unredirected_hdrs)
            self.assertNotIn("Authorization", req.headers)

    def test_sha256_mismatch_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run("https://cdn.example/x", expected_sha256="00" * 32)
        self.assertIn("sha256", str(ctx.exception))
        self.assertEqual(len(self.requests), 1)  # not retried
        self.assertEqual(os.listdir(self.tmp.name), [])  # no dest, no .part

    def test_sha256_match_kept(self):
        import hashlib
        digest = hashlib.sha256(b"abc").hexdigest().upper()
        self._run("https://cdn.example/x", expected_sha256=digest)
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "m.safetensors")))

    def _existing(self, body=b"old"):
        path = os.path.join(self.tmp.name, "m.safetensors")
        with open(path, "wb") as f:
            f.write(body)
        return path

    def test_overwrite_keeps_existing_file_when_download_fails(self):
        path = self._existing()
        with self.assertRaises(RuntimeError):
            self._run("https://cdn.example/x", overwrite=True, expected_sha256="00" * 32)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"old")
        self.assertEqual(os.listdir(self.tmp.name), ["m.safetensors"])

    def test_overwrite_replaces_existing_file_after_download(self):
        path = self._existing()
        self._run("https://cdn.example/x", overwrite=True)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"abc")

    def test_invalid_sha256_means_no_check(self):
        self._run("https://cdn.example/x", expected_sha256="not-a-hash")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "m.safetensors")))

    def _http_error(self, code, url):
        import urllib.error
        return urllib.error.HTTPError(url, code, "no", {}, None)

    def _fail(self, code, url, err_url=None, token="k"):
        err = self._http_error(code, err_url or url)
        with patch.object(dl, "get_civitai_token", return_value=token), \
                patch.object(dl.urllib.request, "urlopen", side_effect=err) as uo:
            with self.assertRaises(RuntimeError) as ctx:
                dl.run_download_url(url, "loras", sync=True)
        return str(ctx.exception), uo.call_count

    def test_civitai_401_403_no_key_message_no_retry(self):
        u = "https://civitai.com/api/download/models/1"
        for code in (401, 403):
            msg, calls = self._fail(code, u, token="")
            self.assertIn("CivitAI needs an API key", msg)
            self.assertEqual(calls, 1)

    def test_civitai_401_403_with_key_says_rejected(self):
        u = "https://civitai.com/api/download/models/1"
        for code in (401, 403):
            msg, calls = self._fail(code, u, token="k")
            self.assertIn("CivitAI rejected your API key", msg)
            self.assertEqual(calls, 1)

    def test_cdn_403_after_redirect_not_reported_as_key_problem(self):
        msg, _ = self._fail(403, "https://civitai.com/api/download/models/1",
                            err_url="https://cdn.example/expired", token="k")
        self.assertNotIn("API key", msg)

    def test_retry_payload_keeps_sha256(self):
        import ComfyUI_HuggingFace_Downloader.web_api as wa
        payload = wa._build_retry_payload({"url": "https://civitai.com/api/download/models/1",
                                           "folder": "loras", "sha256": "ab" * 32})
        self.assertEqual(payload["sha256"], "ab" * 32)
        self.assertNotIn("sha256", wa._build_retry_payload({"url": "https://x/y", "folder": "loras"}))


class TestLookupFixes(unittest.TestCase):
    _run = TestLookup._run

    def _search_with_file(self, name, model_type="LORA", sha="c" * 64):
        search = _fx("models_search_moody.json")
        item = json.loads(json.dumps(search["items"][0]))
        item["id"], item["type"], item["name"] = 77, model_type, "Some LoRA"
        item["modelVersions"] = item["modelVersions"][:1]
        item["modelVersions"][0]["files"] = [{
            "name": name, "sizeKB": 1024, "hashes": {"SHA256": sha.upper()},
            "downloadUrl": "https://civitai.com/api/download/models/777", "type": "Model",
        }]
        search["items"].insert(0, item)
        return search

    def test_model_type_recorded_from_search_by_hash_and_archive(self):
        result, _ = self._run("moodyKrea2MixV9IsHere_v80_int8.safetensors", _routes())
        self.assertEqual(result["exact"]["model_type"], "Checkpoint")
        self.assertTrue(all(c["model_type"] == "Checkpoint" for c in result["family"]))
        result, _ = self._run(RENAMED, _routes(search=False))
        self.assertEqual(result["exact"]["model_type"], "Checkpoint")
        result, _ = self._run(RENAMED, _routes(search=False, by_hash=False))
        self.assertEqual(result["exact"]["model_type"], "Checkpoint")

    def test_generic_name_match_is_family_not_exact(self):
        search = self._search_with_file("pytorch_lora_weights.safetensors")
        result, _ = self._run("pytorch_lora_weights.safetensors",
                              [("/api/v1/models?", search), ("/api/search", {"results": []})])
        self.assertIsNone(result["exact"])
        self.assertIn("pytorch_lora_weights.safetensors", [c["filename"] for c in result["family"]])

    def test_generic_name_match_case_insensitive(self):
        search = self._search_with_file("Model.safetensors")
        result, _ = self._run("model.safetensors",
                              [("/api/v1/models?", search), ("/api/search", {"results": []})])
        self.assertIsNone(result["exact"])

    def test_generic_name_exact_when_archive_confirms_sha(self):
        search = self._search_with_file("model.safetensors", sha="c" * 64)
        archive = {"results": [{"name": "model.safetensors", "url": "/sha256/" + "c" * 64}]}
        result, _ = self._run("model.safetensors", [("/api/v1/models?", search), ("/api/search", archive)])
        self.assertEqual(result["exact"]["match"], "hash")
        self.assertEqual(result["exact"]["sha256"], "c" * 64)

    def test_generic_name_unconfirmed_archive_hash_is_not_exact(self):
        search = self._search_with_file("model.safetensors", sha="c" * 64)
        archive = {"results": [{"name": "model.safetensors", "url": "/sha256/" + HASH}]}
        routes = [("/api/v1/models?", search), ("/api/search", archive),
                  ("by-hash", _fx("model_version_by_hash.json")), ("/api/sha256/", _fx("archive_sha256.json"))]
        result, api = self._run("model.safetensors", routes)
        self.assertIsNone(result["exact"])
        self.assertFalse(any("by-hash" in u or "/api/sha256/" in u for u in api.calls))

    def test_lora_folder_drops_checkpoint_family(self):
        result, _ = self._run("moodyKrea2MixV9IsHere_vUnknown_int8.safetensors", _routes(archive_search=False),
                              folder="loras")
        self.assertIsNone(result)

    def test_diffusion_models_folder_keeps_checkpoint_family(self):
        result, _ = self._run("moodyKrea2MixV9IsHere_vUnknown_int8.safetensors", _routes(archive_search=False),
                              folder="diffusion_models")
        self.assertTrue(result["family"])

    def test_lora_folder_keeps_lora_family_and_unknown_folder_unfiltered(self):
        search = self._search_with_file("someLora_v1.safetensors", model_type="LoCon")
        routes = [("/api/v1/models?", search), ("/api/search", {"results": []})]
        result, _ = self._run("someLora_vUnknown.safetensors", routes, folder="LoRAs")
        self.assertEqual({c["model_type"] for c in result["family"]}, {"LoCon"})
        result, _ = self._run("someLora_vUnknown.safetensors", routes, folder="custom_stuff")
        self.assertIn("Checkpoint", {c["model_type"] for c in result["family"]})

    def test_exact_never_filtered_by_folder(self):
        result, _ = self._run("moodyKrea2MixV9IsHere_v80_int8.safetensors", _routes(), folder="loras")
        self.assertEqual(result["exact"]["match"], "exact")
        self.assertEqual(result["family"], [])

    def test_archive_prefers_non_hf_mirror(self):
        data = _fx("archive_sha256.json")
        hf = [m for m in data["files"] if "huggingface.co" in m["url"]]
        data["files"] = hf + [m for m in data["files"] if m not in hf]
        routes = [("/api/search", _fx("archive_search.json")), ("/api/sha256/", data)]
        result, _ = self._run(RENAMED, routes)
        self.assertNotIn("huggingface.co", result["exact"]["url"])

    def test_archive_hf_only_mirror_still_used(self):
        data = _fx("archive_sha256.json")
        data["files"] = [m for m in data["files"] if "huggingface.co" in m["url"]]
        routes = [("/api/search", _fx("archive_search.json")), ("/api/sha256/", data)]
        result, _ = self._run(RENAMED, routes)
        self.assertIn("huggingface.co", result["exact"]["url"])


class TestAuthWarning(unittest.TestCase):
    def _open_401(self, req, timeout=None):
        raise cd.urllib.error.HTTPError(req.full_url, 401, "x", {}, None)

    def _calls(self, n, url="https://civitai.com/api/v1/models?query=a"):
        import contextlib
        import io
        out = io.StringIO()
        with patch.object(cd, "get_civitai_token", return_value="supersecret"), \
                patch.object(cd.urllib.request, "urlopen", self._open_401), contextlib.redirect_stdout(out):
            for _ in range(n):
                self.assertIsNone(cd._get_json(url, True))
        return out.getvalue()

    def test_logged_once_per_scan_without_key(self):
        cd._reset_auth_warning()
        log = self._calls(3)
        self.assertEqual(log.count("[CivitAI] lookup returned 401/403"), 1)
        self.assertIn("check the CivitAI API key", log)
        self.assertNotIn("supersecret", log)
        cd._reset_auth_warning()
        self.assertEqual(self._calls(1).count("401/403"), 1)

    def test_not_logged_for_archive(self):
        cd._reset_auth_warning()
        self.assertEqual(self._calls(1, "https://civitaiarchive.com/api/search?q=a"), "")

    def test_search_missing_resets_warning(self):
        cd._auth_warned = True
        with patch.object(dl, "civitai_search_enabled", return_value=True), \
                patch.object(cd, "lookup", return_value=None):
            cd.search_missing([{"filename": "a.safetensors"}], skip_all=False, skip_filenames=set(),
                              request_id=None)
        self.assertFalse(cd._auth_warned)


class TestSearchMissingFixes(unittest.TestCase):
    def setUp(self):
        p = patch.object(dl, "civitai_search_enabled", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    _hf_row = TestSearchMissing._hf_row

    def test_rows_without_url_looked_up_first(self):
        order = []

        def fake(name, budget, status_cb=None, **kw):
            order.append(name)
            return None

        rows = [self._hf_row(), {"filename": "plain.safetensors"}]
        with patch.object(cd, "lookup", fake):
            cd.search_missing(rows, skip_all=False, skip_filenames=set(), request_id=None)
        self.assertEqual(order, ["plain.safetensors", "moody_v9.safetensors"])

    def test_hf_row_gets_only_exact_alternative(self):
        rows = [self._hf_row()]
        fam = [_cand(filename="f.safetensors", url="https://civitai.com/d/f", match="family")]
        with patch.object(cd, "lookup", return_value={"exact": _cand(), "family": fam}):
            cd.search_missing(rows, skip_all=False, skip_filenames=set(), request_id=None,
                              hf_sha256=lambda r, p: None)
        urls = [a["url"] for a in rows[0]["alternatives"]]
        self.assertEqual(urls, ["u", "https://civitai.com/api/download/models/1"])

    def test_hf_row_family_only_adds_nothing(self):
        rows = [self._hf_row()]
        fam = [_cand(filename="f.safetensors", url="https://civitai.com/d/f", match="family")]
        with patch.object(cd, "lookup", return_value={"exact": None, "family": fam}):
            cd.search_missing(rows, skip_all=False, skip_filenames=set(), request_id=None)
        self.assertEqual(len(rows[0]["alternatives"]), 1)

    def test_hf_sha_fetch_skipped_when_budget_gone(self):
        def fake(name, budget, status_cb=None, **kw):
            budget.exhaust()
            return {"exact": _cand(), "family": []}

        hf = MagicMock(return_value="a" * 64)
        rows = [self._hf_row()]
        with patch.object(cd, "lookup", fake):
            cd.search_missing(rows, skip_all=False, skip_filenames=set(), request_id=None, hf_sha256=hf)
        hf.assert_not_called()
        label = rows[0]["alternatives"][1]["label"]
        self.assertFalse(label.endswith("same file") or label.endswith("different file"))

    def test_hf_sha_fetch_counts_against_budget(self):
        seen = {}

        def fake(name, budget, status_cb=None, **kw):
            seen["budget"] = budget
            budget.remaining = 1
            return {"exact": _cand(), "family": []}

        rows = [self._hf_row()]
        with patch.object(cd, "lookup", fake):
            cd.search_missing(rows, skip_all=False, skip_filenames=set(), request_id=None,
                              hf_sha256=lambda r, p: "a" * 64)
        self.assertEqual(seen["budget"].remaining, 0)
        self.assertTrue(rows[0]["alternatives"][1]["label"].endswith("same file"))

    def test_folder_passed_to_lookup(self):
        kws = []

        def fake(name, budget, status_cb=None, **kw):
            kws.append(kw.get("folder"))
            return None

        rows = [{"filename": "a.safetensors", "suggested_folder": "loras/sub"},
                {"filename": "b.safetensors", "directory": "vae"},
                {"filename": "c.safetensors"}]
        with patch.object(cd, "lookup", fake):
            cd.search_missing(rows, skip_all=False, skip_filenames=set(), request_id=None)
        self.assertEqual(kws, ["loras", "vae", None])

    def _md_patches(self, skipped, cancelled):
        import ComfyUI_HuggingFace_Downloader.model_discovery as md
        return [patch.object(md, "is_model_skipped", side_effect=skipped),
                patch.object(md, "is_search_cancelled", side_effect=cancelled),
                patch.object(md, "set_current_searching_model")]

    def _run_with(self, patches, rows):
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with patch.object(cd, "lookup", return_value={"exact": _cand(), "family": []}):
            cd.search_missing(rows, skip_all=False, skip_filenames=set(), request_id="r")

    def test_skipped_during_lookup_result_dropped(self):
        rows = [{"filename": "a.safetensors"}]
        self._run_with(self._md_patches([False, True], [False, False]), rows)
        self.assertEqual(rows, [{"filename": "a.safetensors"}])

    def test_cancelled_during_lookup_raises(self):
        import ComfyUI_HuggingFace_Downloader.model_discovery as md
        rows = [{"filename": "a.safetensors"}]
        with self.assertRaises(md.SearchCancelledException):
            self._run_with(self._md_patches([False, False], [False, True]), rows)
        self.assertEqual(rows, [{"filename": "a.safetensors"}])


class TestLabelSize(unittest.TestCase):
    def test_under_one_gb_shows_mb(self):
        self.assertIn("· 512 MB ·", cd._label(_cand(size_bytes=512 * 1024 ** 2)))

    def test_one_gb_and_up_shows_gb(self):
        self.assertIn("· 1.0 GB ·", cd._label(_cand(size_bytes=1024 ** 3)))

    def test_family_labels_distinguishable_by_filename(self):
        a = cd._label(_cand(filename="a.safetensors"))
        b = cd._label(_cand(filename="b.safetensors"))
        self.assertNotEqual(a, b)



TEEKAYS = "Krea2\\A Queen's Breasts\\TeeKaysTittyTime_AQueensTits_V1-3_Kr2.safetensors"


class TestBrandPrefixedNames(unittest.TestCase):
    """Creator brand first, model name after the first underscore."""

    def test_segment_query_after_brand(self):
        qs = cd.search_queries(TEEKAYS)
        self.assertEqual(qs[0], "tee kays titty time a queens tits 3 kr 2")
        self.assertIn("a queens tits", qs)
        self.assertLess(qs.index("a queens tits"), qs.index("tee kays titty time"))

    def test_short_segment_not_queried_alone(self):
        self.assertNotIn("kr 2", cd.search_queries(TEEKAYS))

    def test_no_extra_queries_when_rest_is_noise(self):
        qs = cd.search_queries("moodyKrea2MixUncensoredV8SEGoes_v80_int8.safetensors")
        self.assertEqual(qs, ["moody krea 2 mix uncensored v8se goes", "moody krea 2 mix", "moody krea"])

    def test_keeps_searching_until_exact_file_found(self):
        api = FakeApi([
            ("query=tee%20kays%20titty%20time%20a%20queens", _fx("models_search_teekays_brand.json")),
            ("query=a%20queens%20tits", _fx("models_search_queens.json")),
            ("/api/v1/models?", {"items": []}),
            ("/api/search", {"results": []}),
        ])
        with patch.object(cd, "_get_json", api):
            result = cd.lookup(TEEKAYS, cd.Budget(), folder="loras/Krea2")
        exact = result["exact"]
        self.assertEqual(exact["match"], "exact")
        self.assertEqual(exact["filename"], "TeeKaysTittyTime_AQueensTits_V1-3_Kr2.safetensors")
        self.assertEqual(exact["model_type"], "LORA")
        self.assertTrue(exact["url"].startswith("https://civitai.com/api/download/models/3133173"))
        self.assertFalse(any("/api/search" in u for u in api.calls))  # no archive call needed

    def test_stops_after_exact_hit(self):
        api = FakeApi([("/api/v1/models?", _fx("models_search_queens.json"))])
        with patch.object(cd, "_get_json", api):
            cd.lookup("TeeKaysTittyTime_AQueensTits_V1-3_Kr2.safetensors", cd.Budget())
        self.assertEqual(len(api.calls), 1)

if __name__ == "__main__":
    unittest.main()
