# CivitAI + CivitAI Archive Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Missing workflow models the Hugging Face search can't resolve are looked up on CivitAI and civitaiarchive.com, and offered in the existing download prompt (exact match, renamed-by-hash match, or family alternatives).

**Architecture:** A new stdlib-only module `civitai_discovery.py` owns all CivitAI/archive HTTP and matching. `model_discovery.process_workflow_for_missing_models` calls it once after the HF stages. Results reuse the existing row fields (`url`, `source`, `alternatives`) plus a new `sha256`; the download path (`downloader.run_download_url`) gains CivitAI auth and sha256 verification. The JS only learns to show `alt.label` and pass `sha256` through.

**Tech Stack:** Python 3.12 stdlib (`urllib`, `hashlib`, `json`, `re`), ComfyUI frontend extension JS, `unittest` run by pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-civitai-search-design.md`

## Global Constraints

- No new Python dependencies; HTTP via `urllib.request` only.
- CivitAI API host constant: `CIVITAI_API = "https://civitai.com"`; key sent only to hosts in `CIVITAI_HOSTS = {"civitai.com", "civitai.red", "civitai.green"}` (and their `www.`), via `add_unredirected_header`. Never to civitaiarchive.com or mirrors.
- Archive base: `ARCHIVE_API = "https://civitaiarchive.com"`.
- Settings ids: `downloader.civitai_token` (text, default `""`), `downloader.civitai_search_enabled` (boolean, default `true`). Key fallback: `<custom_nodes>/comfyui-lora-manager/settings.json` → `civitai_api_key`.
- Status `source` values: `civitai_search`, `civitai_hash`, `civitai_archive`.
- Only sha256-proven or exact-filename matches keep the workflow's filename; family alternatives use their real CivitAI filename.
- HF stays the row's primary `url` when present; CivitAI becomes an alternative on that row.
- CivitAI lookups honour `skip_hf_search`, `skip_filenames` and cancel exactly like HF.
- Budget: `CIVITAI_MAX_REQUESTS = 40` per scan, `CIVITAI_DEADLINE_S = 45`; stop on HTTP 429.
- Max family alternatives per row: 8.
- Tests make no network calls (patch `civitai_discovery._get_json`).
- Commits: conventional (`feat:`/`fix:`/`test:`/`docs:`), git identity only, **no AI/Claude attribution, no session links, no Co-Authored-By**.
- Push only to remote `fork` (never `origin`).

## Review Focus

1. Filename with path prefix (`Krea2/foo.safetensors`) or Windows backslashes → match on basename only. (Task 2 test)
2. CivitAI returns 401/403 for a search or by-hash (no/invalid key) → lookup degrades to archive, scan never errors. (Task 3 test)
3. API returns malformed JSON / missing keys (`files` absent, `hashes` null) → candidate skipped, no exception. (Task 3 test)
4. Download redirects from civitai.com to a CDN host → Authorization not present on the redirected request. (Task 5 test)
5. Model already on disk → no CivitAI request at all. (Task 4 test: `search_missing` not called when `missing` is empty)

---

### Task 0: Fork, branch, commit existing work

**Files:** existing uncommitted `model_discovery.py` lock fix; `docs/superpowers/specs/…`, `docs/superpowers/plans/…`

- [ ] **Step 1:** `gh repo fork jnxmx/ComfyUI_HuggingFace_Downloader --remote=false --clone=false`; `git remote add fork git@github.com:vitosans/ComfyUI_HuggingFace_Downloader.git`; `git fetch origin`; `git switch -c civitai-search origin/main` (working-tree changes carry over).
- [ ] **Step 2:** Run `../../venv/bin/python -m pytest tests -q` → `14 passed`.
- [ ] **Step 3:** Commit `model_discovery.py` alone: `fix: guard widget schema hint cache with a lock`.
- [ ] **Step 4:** Commit docs: `docs: design and plan for CivitAI model lookup`.

### Task 1: Settings + CivitAI key loader

**Files:**
- Modify: `js/settings.js` (add two settings after `downloader.hf_token`, category `["Hugging Face Download & Backup", "Tokens", "CivitAI API key"]` and `[..., "Auto download", "Search CivitAI for missing models"]`)
- Modify: `downloader.py` (after `get_token`)
- Test: `tests/test_civitai.py`

**Interfaces:**
- Produces: `downloader.get_civitai_token() -> str | None`; `downloader.civitai_search_enabled() -> bool`.

- [ ] **Step 1: Failing tests** (`TestCivitaiSettings`, same sys.modules mocking header as `tests/test_issue_4_subfolder_models.py`):
  - `test_token_from_settings`: patch `downloader._load_comfy_settings_dict` → `{"downloader.civitai_token": " abc "}`; assert `get_civitai_token() == "abc"`.
  - `test_token_falls_back_to_lora_manager`: settings `{}`, patch `downloader._lora_manager_settings_path` to a temp file containing `{"civitai_api_key": "lm"}`; assert `== "lm"`.
  - `test_token_none_when_absent`: both empty, `CIVITAI_API_TOKEN` unset → `None`.
  - `test_search_enabled_default_true` / `test_search_disabled`: `{}` → True; `{"downloader.civitai_search_enabled": False}` → False.
- [ ] **Step 2:** Run `pytest tests/test_civitai.py -q` → FAIL (AttributeError).
- [ ] **Step 3:** Implement. Order: setting → env `CIVITAI_API_TOKEN` → LoRA Manager file. `_lora_manager_settings_path() -> str` = `<dir of this repo>/../comfyui-lora-manager/settings.json`. Bad/missing file → skip silently.
- [ ] **Step 4:** Tests PASS.
- [ ] **Step 5:** Commit `feat: CivitAI API key and search toggle settings`.

### Task 2: Filename → search words

**Files:** Create `civitai_discovery.py`; Test `tests/test_civitai.py`

**Interfaces:**
- Produces: `search_queries(filename: str) -> list[str]` (most specific first, deduped, max 3, each ≥ 2 words or ≥ 5 chars, else dropped).

- [ ] **Step 1: Failing tests** (`TestSearchQueries`):
  - `search_queries("moodyKrea2MixUncensoredV8SEGoes_v80_int8.safetensors")[0] == "moody krea 2 mix uncensored v8se goes"` and the list contains `"moody krea 2 mix"`.
  - `search_queries("Krea2\\sub/foo_bar_v1.safetensors")[0] == "foo bar"` (basename, backslashes).
  - Precision/format tokens dropped: `search_queries("cyberrealistic_v90_fp8_e4m3fn.gguf")[0] == "cyberrealistic"`.
  - `search_queries("x.safetensors") == []`.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement: basename, strip extension, split on `_`/`-`/`.`/space; drop the trailing version/precision tokens matching `^(v\d+(\.\d+)?|fp\d+.*|bf16|int\d|q\d.*|e\dm\d.*|gguf|pruned|ema(only)?)$` (case-insensitive) only from the segment *after* the first `_`; split camelCase and letter↔digit boundaries except a letter+digits+letters run like `V8SE` (keep `v8se`); lowercase. Queries: full words, then first 4 words, then first 2 words.
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: derive CivitAI search words from model filenames`.

### Task 3: CivitAI / archive lookup

**Files:** Modify `civitai_discovery.py`; Test `tests/test_civitai.py` (fixtures in `tests/fixtures/civitai/*.json`, trimmed copies of the live responses: `models_search_moody.json`, `model_version_by_hash.json`, `archive_search.json`, `archive_sha256.json`)

**Interfaces:**
- Consumes: `search_queries`, `downloader.get_civitai_token`.
- Produces:
  - `_get_json(url: str, auth: bool) -> dict | None` — the only network call; returns None on any HTTP/JSON error; raises `RateLimited` on 429.
  - `class Budget(max_requests: int = CIVITAI_MAX_REQUESTS, deadline_s: float = CIVITAI_DEADLINE_S)` with `.spend() -> bool` (False once exhausted).
  - `lookup(filename: str, budget: Budget, status_cb=None) -> LookupResult | None` where
    `LookupResult = {"exact": Candidate | None, "family": list[Candidate]}` and
    `Candidate = {"filename": str, "url": str, "sha256": str | None, "size_bytes": int | None, "base_model": str | None, "version_name": str, "model_name": str, "nsfw": bool, "site": "civitai" | "civitai_archive", "match": "exact" | "hash" | "family", "civitai_type": str | None}`.

- [ ] **Step 1: Failing tests** (`TestLookup`, patch `_get_json` with a dict url-substring → fixture):
  - `test_exact_filename_in_family`: wanted `moodyKrea2MixV9IsHere_v80_3213132.safetensors` → `exact.match == "exact"`, `exact.url` starts `https://civitai.red/api/download/models/3327244`, `exact.sha256` lowercase hex.
  - `test_renamed_file_found_by_hash`: wanted `moodyKrea2MixUncensoredV8SEGoes_v80_int8.safetensors` (search fixture has no such name) → `exact.match == "hash"`, `exact.sha256 == "7b9c56858a518f9fbed261aa8c535d3bb409fc8ecf441dad780902703c1ed454"`, `exact.filename` is the *live* name.
  - `test_archive_mirror_when_civitai_lost_it`: by-hash returns None → `exact.site == "civitai_archive"`, url = first mirror with `deletedAt is None and not is_gated and not is_paid`.
  - `test_family_same_base_only`: no exact; family items all have `base_model == "Krea 2"`, other-model entries with another base excluded, ≤ 8, same-model files first.
  - `test_family_unknown_base_same_model_only`: archive gives no base, search returns 2 models → only files of the top model.
  - `test_unauthorized_degrades`: `_get_json` returns None for civitai URLs → falls through to archive; no exception.
  - `test_malformed_payload_skipped`: model with `modelVersions: [{"files": null}]` → no exception, `lookup` returns family from other models or None.
  - `test_budget_stops_requests`: `Budget(max_requests=1)` → `_get_json` called once.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement. Order: (a) for each query: `GET {CIVITAI_API}/api/v1/models?query=<q>&limit=5&nsfw=true` (auth), scan `items[].modelVersions[].files[]` for basename equal (case-insensitive) → exact; remember items for family; stop at first query with items. (b) if no exact: `GET {ARCHIVE_API}/api/search?q=<basename>`, pick result whose `name` equals the basename and `url` starts `/sha256/`; take hash; `GET {CIVITAI_API}/api/v1/model-versions/by-hash/<hash>` (auth), file whose `hashes.SHA256.lower() == hash` → `match="hash"`, base from `baseModel`; (c) else `GET {ARCHIVE_API}/api/sha256/<hash>` → mirror → `site="civitai_archive"`, `match="hash"`. Family: same-model files (other versions) first, then other models from the search whose version `baseModel` equals the known base; none from other models if base unknown. `size_bytes = round(sizeKB*1024)`. Each `_get_json` call is preceded by `budget.spend()`; status_cb called with the three `source` values.
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat: look up missing models on CivitAI and CivitAI Archive`.

### Task 4: Pipeline integration

**Files:** Modify `civitai_discovery.py` (row merge), `model_discovery.py` (call after `_run_hf_stage`, before step "6. Quantized variant detection"); Test `tests/test_civitai.py`

**Interfaces:**
- Consumes: `lookup`, `Budget`, `civitai_search_enabled`, `is_search_cancelled`, `is_model_skipped`.
- Produces: `search_missing(models: list[dict], *, skip_all: bool, skip_filenames: set[str], request_id: str | None, status_cb=None, hf_sha256=None) -> None` (mutates rows in place, matching the existing pipeline style); `hf_sha256: Callable[[str, str], str | None]` defaulting to `downloader.get_remote_file_metadata`-based lookup of `(hf_repo, hf_path)`.

Row rules:
- Row without `url` + `exact` → `url`, `source="civitai"`, `sha256`, keep `filename`; `family` → `alternatives` (append after any existing).
- Row with HF `url` (`_is_huggingface_url`-equivalent: host `huggingface.co`) + `exact` → append one alternative; primary unchanged.
- Row with non-HF `url` → untouched.
- Alternative dict: `{"filename", "url", "sha256", "source": "civitai" | "civitai_archive", "label", "suggested_folder"?}`; `label = "CivitAI · <model_name> <version_name> · <X.X GB> · <base_model>"` + `" · NSFW"` if nsfw + for the HF-row case `" · same file"` / `" · different file"` when both sha256 known.

- [ ] **Step 1: Failing tests** (`TestSearchMissing`, patch `civitai_discovery.lookup`):
  - `test_unresolved_row_gets_civitai_url` (filename unchanged, `sha256` set, `source == "civitai"`).
  - `test_hf_row_keeps_hf_primary_and_gets_alternative` with `hf_sha256` returning the same hash → alt label endswith `"same file"`; different → `"different file"`.
  - `test_family_alternatives_keep_real_filename`.
  - `test_skip_all_makes_no_lookup` / `test_skip_filename_respected` / `test_disabled_setting_makes_no_lookup`.
  - `test_nothing_missing_no_lookup`: call `process_workflow_for_missing_models` path with all models present (patch `check_model_files` → `([], found, [])`) → `search_missing` not called.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement `search_missing`; in `model_discovery.py` add one call `civitai_discovery.search_missing(missing_models, skip_all=skip_hf_search_all, skip_filenames=skip_filenames, request_id=request_id, status_cb=status_cb)` guarded by `if missing_models:` and wrapped in `try/except Exception` that logs `[CivitAI]` and continues (a lookup bug must never break the scan). Catch `SearchCancelledException` re-raise.
- [ ] **Step 4:** Full suite PASS.
- [ ] **Step 5:** Commit `feat: offer CivitAI matches and alternatives for missing models`.

### Task 5: Download auth + sha256 verification

**Files:** Modify `downloader.py` (`run_download_url`), `web_api.py` (pass `item.get("sha256")` at the three `run_download_url(` call sites: lines ~2086, ~2279/2320 if they are file downloads, ~2641); Test `tests/test_civitai.py`

**Interfaces:**
- Produces: `run_download_url(..., expected_sha256: Optional[str] = None)`.

- [ ] **Step 1: Failing tests** (`TestDirectDownload`, patch `urllib.request.urlopen` / `build_opener` and `resolve_target_dir` → temp dir, `is_path_within_allowed_roots` → True):
  - `test_civitai_request_has_unredirected_auth`: with token "k", request to `https://civitai.com/api/download/models/1` has `Authorization: Bearer k` in `req.unredirected_hdrs` and not in `req.headers`.
  - `test_non_civitai_has_no_auth`: `https://civitaiarchive.com/...` and `https://cdn.example/...` → no Authorization anywhere.
  - `test_sha256_mismatch_rejected`: body `b"abc"`, expected `"00"*32` → raises/returns failure and destination file does not exist.
  - `test_sha256_match_kept`: expected = sha256(b"abc") → file exists.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3:** Implement: host check on `urlparse(raw_url).hostname` (strip `www.`) ∈ `CIVITAI_HOSTS` and token present → `request.add_unredirected_header("Authorization", f"Bearer {token}")`. Hash while streaming to the `.part` file; compare before the atomic rename; on mismatch delete `.part` and raise `RuntimeError("Downloaded file failed sha256 check")` (not retried). On HTTP 401/403 from a civitai host raise `RuntimeError("CivitAI needs an API key for this file — add it in Settings → Hugging Face Download & Backup → CivitAI API key")`.
- [ ] **Step 4:** PASS (full suite).
- [ ] **Step 5:** Commit `feat: CivitAI download auth and sha256 verification`.

### Task 6: Frontend: labels + sha256 passthrough

**Files:** Modify `js/auto_download.js` (~3466 alt label, ~3485 `useBtn.onclick`, row construction where `rowData` gets `resolvedUrl`, ~3768 `toDownload` item)

- [ ] **Step 1:** Alt label: `altLabel.textContent = alt.label || \`${alt.filename}${alt.source ? " • " + alt.source : ""}\``.
- [ ] **Step 2:** `rowData.sha256 = m.sha256 || ""`; in `useBtn.onclick` set `rowData.sha256 = alt.sha256 || ""`; in `toDownload` add `sha256: r.sha256 || ""`. If the user edits the URL input manually, clear `rowData.sha256` (input `change` listener).
- [ ] **Step 3:** `node --check js/auto_download.js` → no output.
- [ ] **Step 4:** Manual check in ComfyUI (restart ComfyUI in tmux session `comfy`, hard-refresh browser): open a workflow referencing `moodyKrea2MixUncensoredV8SEGoes_v80_int8.safetensors`, open the downloader's missing-model search: row shows a CivitAI URL; Alternatives list shows labelled CivitAI options; queue nothing.
- [ ] **Step 5:** Commit `feat: show CivitAI options in the missing-model prompt`.

### Task 7: Docs + push

- [ ] **Step 1:** README: add a "CivitAI lookup" section (what it searches, key setting + LoRA Manager fallback, off switch). Bump `pyproject.toml` version to `1.34.24+civitai.1`.
- [ ] **Step 2:** Full suite PASS; `git log origin/main..` shows only conventional commits with no attribution lines.
- [ ] **Step 3:** Commit `docs: document CivitAI lookup`; `git push -u fork civitai-search`.
