# CivitAI + CivitAI Archive lookup for missing models — design

Fork: `vitosans/ComfyUI_HuggingFace_Downloader`, branch `civitai-search` (upstream `jnxmx/ComfyUI_HuggingFace_Downloader`, v1.34.24).
Personal fork for one user's machine; keep the diff mergeable with upstream so upstream updates can still be pulled.

Git setup: the local checkout is on `main` with `origin` = upstream. Create the fork with `gh repo fork --remote=false`,
add it as remote `fork`, branch `civitai-search` from `origin/main`, and push **only to `fork`**. Never push to `origin`.

## Why

The node already scans a workflow on open/Run, finds models that aren't on disk, searches Hugging Face, and
shows a prompt where the user picks and starts downloads. Many of this user's workflows use CivitAI models that
Hugging Face never has, so they show up as "not found". Example from the real log:
`moodyKrea2MixUncensoredV8SEGoes_v80_int8.safetensors`.

Facts verified against the live APIs (2026-11-04):

- **civitai.com / civitai.green / civitai.red share one catalogue and API.** `civitai.green/api/...` 301-redirects to
  `civitai.com`; `.com` and `.red` return identical search results with `nsfw=true`. One API host is enough.
- **Searching by raw filename finds nothing**; searching by the words in the name does
  (`query=moody krea 2 mix` → model 2731187). The API's `query` matches model *names*, not filenames.
- **CivitAI renames files** when a creator renames the model. The workflow's file is now served as
  `moodyKrea2MixV9IsHere_v80BF16FP8_int8.safetensors`. A filename-only match can never find it.
- **civitaiarchive.com maps old filenames to sha256.**
  `GET https://civitaiarchive.com/api/search?q=<filename>` → `results[].name`, `results[].url = /sha256/<hash>`, `base_model`, `is_deleted`.
  `GET https://civitaiarchive.com/api/sha256/<hash>` → `files[].filename/url`, `model.name`, `model.version.{name,baseModel,platform_url}`,
  `model.version.files[].{name,sizeKB,sha256,downloadUrl,mirrors[].{url,source,deletedAt,is_gated,is_paid}}`.
- **CivitAI resolves a sha256**: `GET /api/v1/model-versions/by-hash/<sha256>` returns the live version + files.
- **CivitAI file records carry sha256** (`files[].hashes.SHA256`); Hugging Face LFS files carry sha256 too.

## Goals

1. When a missing model isn't resolved by the existing Hugging Face logic, look it up on CivitAI, then CivitAI Archive.
2. **Exact match** (same filename, or same file by sha256 under a newer name): offer it as the model's download source.
   A sha256 match is saved under the filename the workflow expects, so the workflow loads unchanged.
3. **No exact match but the family is found**: offer options in the existing **Alternatives (N)** list:
   - first, other files/versions of the same CivitAI model (other versions, BF16/FP8/int8/GGUF builds);
   - then closely named models on the **same base model** (e.g. Krea 2); other base models are not offered *(Claude's call — user may want this widened)*.
     The base model is taken from the same-model family or the archive's `base_model`. **If it is unknown, offer only
     the same-model family, never other models** *(Claude's call)*.
   Picking an alternative keeps its real CivitAI filename and the existing "Use" + post-download widget update points the workflow at it.
4. **Found on both Hugging Face and CivitAI**: show both sources for that row; **Hugging Face is preselected** (user decision).
   Show size per source, and when both sha256 are known, label "same file" vs "different file".
5. Each CivitAI/archive option shows: version name, size, base model, source site, NSFW tag.
6. Nothing downloads without the user's click — same prompt, same flow as today (user decision).

## Lookup order (per missing model, stop at first exact match)

1. Existing Hugging Face logic, unchanged.
2. **CivitAI search**: split the filename stem into words (camelCase, digits, `_`/`-`; drop version/precision
   tokens like `v80`, `int8`, `fp8`, `bf16`, `gguf`, `q4_k_m`), query `/api/v1/models?query=<words>&nsfw=true`,
   try progressively shorter word prefixes if empty. Exact = a file in any version whose name equals the wanted filename
   (case-insensitive). Collect the rest as family candidates.
3. **Archive → hash**: `api/search?q=<filename>`; for an exact-name result take its sha256, then
   CivitAI `by-hash/<sha256>`; pick the file whose SHA256 equals it. That's an exact match (renamed file).
4. **Archive mirror**: if CivitAI no longer serves that hash, use `api/sha256/<hash>` mirrors that are not deleted,
   not gated, not paid. Still an exact match.
5. Steps 2–4 also run when the row already has a Hugging Face URL (from HF search, the popular registry, the Manager
   list or the workflow itself), **only to fill the "both sources" picker**, inside the CivitAI budget; HF stays preselected.
   Rows whose URL is not Hugging Face are left alone *(Claude's call)*.
6. CivitAI/archive lookups obey the same switches as HF search: `skip_hf_search` (workflow-open/Run preflights),
   per-file skips and "Skip all unresolved" skip **all** remote lookups. So CivitAI results appear in the full search
   dialog, exactly where HF results do today *(Claude's call — matches "just like this one does")*.

## Constraints

- **Downloads verify sha256** when the source gave one; mismatch = failed download, file not kept.
- **CivitAI API key**: new setting `downloader.civitai_token` ("CivitAI API key") next to the HF token. If empty,
  fall back to `custom_nodes/comfyui-lora-manager/settings.json` → `civitai_api_key` (already set on this machine).
  Sent as `Authorization: Bearer` to civitai hosts only — never to civitaiarchive.com or other mirrors. CivitAI downloads
  redirect to a CDN; add the header with `add_unredirected_header` (or strip it on redirect) so the key never follows.
  Many CivitAI downloads 401 without it; surface "add your CivitAI key" like the existing gated-HF hint.
- **API host**: `civitai.com` *(Claude's call; LoRA Manager uses `civitai.red`, same API)*. Make it a constant.
- **Don't break HF behaviour or its search budget.** CivitAI/archive get their own small budget (time + request
  count) and back off on 429 like the HF code (`is_rate_limited_error`).
- **Status updates**: reuse `status_cb` with new `source` values (`civitai_search`, `civitai_hash`, `civitai_archive`)
  so the existing "Searching…" panel shows progress; cancel/skip must work the same.
- **Off switch**: setting `downloader.civitai_search_enabled`, default on.
- Python stdlib `urllib` for HTTP (as `downloader.py` already does); no new dependencies.
- Keep the existing uncommitted local fix in `model_discovery.py` (lock around the widget-schema cache) — commit it first, separately.
- Commits: conventional format (`feat:`, `fix:` …), **no AI/Claude attribution of any kind**.

## Non-goals

- Browsing CivitAI, a CivitAI model explorer, or the backup features.
- Auto-downloading without a click.
- Searching other sites (Tensor.art, ModelScope, …).
- Pushing to upstream / opening a PR there.

## Anti-goals (would count as failure even if it "works")

- Silently downloading a different model under the old filename (only sha256-proven matches may keep the old name).
- Sending the CivitAI key anywhere except CivitAI.
- Making workflow open noticeably slower when everything is already on disk (no lookups happen then).

## Left to the builder

- Where the code lives (suggested: new `civitai_discovery.py`, called from `process_workflow_for_missing_models` after the HF stage).
- How a row stores multiple sources (extend the existing `alternatives` shape vs. a new `sources` list) and how the
  JS renders the HF/CivitAI picker in `js/auto_download.js`.
- Ranking of family candidates, how many to show (suggest ≤8), and word-splitting details.
- Choosing the destination folder for CivitAI results: reuse the node's existing folder resolution; CivitAI file
  `type` (Checkpoint / Diffusion Model / LORA / VAE) is only a hint.
- Tests: pytest under `tests/` with recorded API responses (no network in tests); cover word-splitting, exact/hash/family
  matching, HF+CivitAI merge with HF preselected, key never sent to non-civitai hosts, sha256 mismatch rejected.
