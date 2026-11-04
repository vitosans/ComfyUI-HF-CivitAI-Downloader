# Hugging Face Download & Backup

ComfyUI node pack and UI tools for:

- downloading models and folders from Hugging Face
- downloading direct model files from non-HF HTTP(S) URLs
- auto-detecting missing workflow models and queueing downloads
- backing up and restoring ComfyUI content to/from a Hugging Face repo

## What This Pack Includes

- 2 ComfyUI nodes:
  - `Hugging Face Download Model`
  - `Hugging Face Download Folder`
- top-menu integration with:
  - `Auto-download models`
  - `Model Explorer`
  - `Download new model`
  - `Backup ComfyUI to Hugging Face`
- queued downloader with cancel + status panel
- missing-model dialog integration (`Auto-search and download missing models`)
- selective backup/restore/delete browser for HF backups

## Core Features

### 1) Direct Hugging Face Download Nodes

- Parse `resolve`, `blob`, `tree`, and shorthand HF links.
- Download a single file or a full folder/subfolder.
- Output `model name` / `folder name` for downstream node wiring.
- Choose standard model folders or custom paths.
- Manual `Download new model` also supports direct non-HF HTTP(S) file URLs.

### 2) Workflow Auto-Discovery + Missing Model Resolution

- Scans workflow nodes and nested subgraphs.
- Detects:
  - found local models
  - path mismatches (`Fix Path`)
  - missing models
- Resolves model URLs using:
  - workflow metadata
  - local curated `metadata/popular-models.json`
  - ComfyUI-Manager model lists/cache when available
  - Hugging Face search fallback
- Suggests compatible alternatives for unresolved quantized variants.

### 3) Background Queue + Status Panel

- Queue selected/manual downloads through `/queue_download`.
- Poll live progress via `/download_status`.
- Cancel queued/active jobs via `/cancel_download`.
- Deferred integrity verification runs after queue idle.
- Refreshes ComfyUI model dropdowns after successful downloads.

### 4) Backup and Restore to Hugging Face

- Backup selected ComfyUI folders/files into a HF repo under `ComfyUI/*`.
- Browse remote backup tree from UI.
- Restore all or selected content.
- Delete selected backup entries from the remote repo.
- Handles custom node snapshots with `comfy-cli` (`custom_nodes_snapshot.yaml`).
- Strips `downloader.hf_token` and `downloader.civitai_token` from backed-up settings and preserves the local tokens on restore.

#### CivitAI lookup

After the Hugging Face search, models still missing are looked up on CivitAI (civitai.com API; civitai.green/.red share it) by the words in the filename, then on civitaiarchive.com by old filename, then sha256, then CivitAI by-hash, then archive mirrors.

- Exact or sha256-proven matches keep the workflow filename. Generic names (`model.safetensors`, `pytorch_lora_weights.safetensors`, ...) only count as exact when the archive confirms the sha256.
- Otherwise same-model/same-base "family" options appear under Alternatives, labelled with model, version, filename, size, base model and NSFW. Family options of a CivitAI type that does not fit the target folder (e.g. a Checkpoint for `loras`) are dropped.
- Rows Hugging Face already resolved only get the exact CivitAI match as an alternative, marked as the same or a different file.
- When both exist, Hugging Face stays preselected. Nothing downloads without a click.
- Downloads with a known sha256 are verified and rejected on mismatch.
- The API key is only sent to CivitAI hosts and is not forwarded on redirects.
- Budget: 40 requests / 45 s per scan. Lookups follow the same skip switches as the Hugging Face search.

## Web API Endpoints

- `GET /folder_structure`
- `POST /check_missing_models`
- `POST /install_models`
- `POST /queue_download`
- `POST /cancel_download`
- `GET /download_status`
- `GET /search_status`
- `GET /model_library`
- `GET /api/model_explorer/categories`
- `GET /api/model_explorer/filters`
- `GET /api/model_explorer/groups`
- `POST /api/model_explorer/download`
- `POST /api/model_explorer/use`
- `POST /api/model_explorer/delete`
- `GET /backup_browser_tree`
- `POST /backup_to_hf`
- `POST /backup_selected_to_hf`
- `POST /restore_from_hf`
- `POST /restore_selected_from_hf`
- `POST /delete_from_hf_backup`
- `POST /restart`

## Settings

Configured in ComfyUI settings:

- `downloader.hf_token` (Hugging Face token)
- `downloader.auto_open_missing_models_on_run` (after Run, auto-opens downloader only when native ComfyUI missing-models dialog appears)
- `downloader.civitai_token` (CivitAI API key; falls back to env `CIVITAI_API_TOKEN`, then ComfyUI-Lora-Manager's saved key)
- `downloader.civitai_search_enabled` (CivitAI lookup for missing models, default on)
- `downloaderbackup.repo_name` (target HF repo for backup/restore)
- `downloaderbackup.file_size_limit` (max single file size in GB for backup)

Model Explorer is always enabled and has no separate toggle in settings.

Environment variables:

- `HF_TOKEN`
- `HF_SEARCH_MAX_CALLS` (default `200`)
- `HF_SEARCH_RATE_LIMIT_SECONDS` (default `300`)
- `HF_SEARCH_MAX_SECONDS` (default `60`)
- `HF_SEARCH_CALL_TIMEOUT` (default `20`)
- `HF_PRIORITY_REPO_SCAN_LIMIT` (default `100`)
- `HF_URL_CHECK_TIMEOUT` (default `8`)
- `HF_DOWNLOADER_SHA_MAX_BYTES` (hash verification cap)

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Restart ComfyUI.

## Notes and Current Limits

- Queue download flow is hybrid for file mode:
  - Hugging Face links use the Hugging Face engine.
  - Non-HF `http(s)` file URLs use direct streaming download.
- Folder/full-repo mode remains Hugging Face-only.
- For gated repos, set a valid token via `downloader.hf_token` or `HF_TOKEN`.
