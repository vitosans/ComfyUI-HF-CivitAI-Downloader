"""CivitAI lookup for models that were not found on Hugging Face (stdlib only)."""
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .downloader import ARCHIVE_API, CIVITAI_API, CIVITAI_HOSTS, get_civitai_token

CIVITAI_MAX_REQUESTS = 40
CIVITAI_DEADLINE_S = 45
_HTTP_TIMEOUT_S = 20
_MAX_FAMILY = 8
_MODEL_EXTENSIONS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft")
_USER_AGENT = "ComfyUI_HuggingFace_Downloader"

# File stems shared by unrelated models: a name match alone is not proof.
_GENERIC_STEMS = frozenset({
    "model", "model.fp16", "pytorch_model", "pytorch_lora_weights", "diffusion_pytorch_model",
    "diffusion_pytorch_model.fp16", "adapter_model", "lora", "unet", "vae", "ae", "clip",
    "text_encoder", "weights", "checkpoint",
})

# Row folder (top segment) -> CivitAI model types that belong there.
_FOLDER_TYPES = {
    "loras": {"lora", "locon", "dora", "lycoris"},
    "lycoris": {"lora", "locon", "dora", "lycoris"},
    "vae": {"vae"},
    "checkpoints": {"checkpoint"},
    "diffusion_models": {"checkpoint"},
    "unet": {"checkpoint"},
    "embeddings": {"textualinversion"},
    "controlnet": {"controlnet"},
    "upscale_models": {"upscaler"},
}

_MAX_QUERIES = 4
_MIN_SEGMENT_LETTERS = 6  # a segment after the brand must carry this many letters/digits to be queried alone
_MIN_QUERY_CHARS = 5
_MIN_QUERY_WORDS = 2

# Trailing version / precision / format markers that never help a text search.
_NOISE_TOKEN = re.compile(
    r"^(v\d+(\.\d+)?|fp\d+.*|bf16|int\d|q\d.*|e\dm\d.*|gguf|pruned|ema(only)?)$",
    re.IGNORECASE,
)
# Word tokens: a mixed run like V8SE stays whole; otherwise camelCase and
# letter/digit boundaries split.
_WORD = re.compile(r"[A-Z]\d+[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_SEPARATORS = re.compile(r"[-.\s]+")


def _words(text):
    return [w.lower() for w in _WORD.findall(text)]


def search_queries(filename):
    """Return up to 4 search strings for ``filename``, most specific first.

    Creators often prefix files with their brand (``TeeKaysTittyTime_AQueensTits``)
    while the CivitAI model is named after the part behind it, so each meaningful
    segment after the first ``_`` is also queried on its own.
    """
    base = re.split(r"[\\/]", filename or "")[-1]
    stem = os.path.splitext(base)[0]
    first, *rest = stem.split("_")

    words = []
    for piece in _SEPARATORS.split(first):
        words.extend(_words(piece))
    segment_queries = []
    for segment in rest:
        if _NOISE_TOKEN.match(segment):
            continue
        segment_words = []
        for piece in _SEPARATORS.split(segment):
            if piece and not _NOISE_TOKEN.match(piece):
                segment_words.extend(_words(piece))
        words.extend(segment_words)
        if len("".join(segment_words)) >= _MIN_SEGMENT_LETTERS:
            segment_queries.append(" ".join(segment_words))

    candidates = [" ".join(words)] + segment_queries + [" ".join(words[:4]), " ".join(words[:2])]
    queries = []
    for query in candidates:
        if (
            query
            and query not in queries
            and (len(query.split()) >= _MIN_QUERY_WORDS or len(query) >= _MIN_QUERY_CHARS)
        ):
            queries.append(query)
    return queries[:_MAX_QUERIES]


class RateLimited(Exception):
    """HTTP 429 from CivitAI or the archive."""


class _Stop(Exception):
    """Internal: budget exhausted or rate limited; abandon the lookup."""


class Budget:
    """Request and wall-clock allowance for one scan."""

    def __init__(self, max_requests=CIVITAI_MAX_REQUESTS, deadline_s=CIVITAI_DEADLINE_S):
        self.remaining = max_requests
        self.deadline = time.monotonic() + deadline_s

    def exhaust(self):
        self.remaining = 0

    def spend(self):
        if self.remaining <= 0 or time.monotonic() >= self.deadline:
            return False
        self.remaining -= 1
        return True


def _host(url):
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


_auth_warned = False


def _reset_auth_warning():
    global _auth_warned
    _auth_warned = False


def _warn_auth_rejected():
    global _auth_warned
    if not _auth_warned:
        _auth_warned = True
        print("[CivitAI] lookup returned 401/403 \u2014 check the CivitAI API key")


def _get_json(url, auth=False):
    """GET ``url`` as JSON. None on any error; RateLimited on HTTP 429."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    if auth and _host(url) in CIVITAI_HOSTS:
        token = get_civitai_token()
        if token:
            req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RateLimited() from e
        if e.code in (401, 403) and _host(url) in CIVITAI_HOSTS:
            _warn_auth_rejected()
        return None
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None


def _basename(filename):
    return re.split(r"[\\/]", filename or "")[-1]


def _list(value):
    return value if isinstance(value, list) else []


def _dict(value):
    return value if isinstance(value, dict) else {}


def _sha(file):
    value = _dict(_dict(file).get("hashes")).get("SHA256")
    return value.lower() if isinstance(value, str) and value else None


def _candidate(file, version, model_name, model_id, nsfw, match, site="civitai", url=None, model_type=None):
    size_kb = file.get("sizeKB")
    return {
        "filename": file["name"],
        "url": url or file.get("downloadUrl") or version.get("downloadUrl") or "",
        "sha256": _sha(file),
        "size_bytes": round(size_kb * 1024) if isinstance(size_kb, (int, float)) else None,
        "base_model": version.get("baseModel") or None,
        "version_name": version.get("name") or "",
        "model_name": model_name or "",
        "nsfw": bool(nsfw),
        "site": site,
        "match": match,
        "civitai_type": file.get("type") if isinstance(file.get("type"), str) else None,
        "model_type": model_type if isinstance(model_type, str) else None,
        "_model_id": model_id,
    }


def _public(candidate):
    return {k: v for k, v in candidate.items() if not k.startswith("_")}


def _model_files(model):
    """Yield (file, version) for downloadable model files of a search item."""
    for version in _list(_dict(model).get("modelVersions")):
        version = _dict(version)
        for file in _list(version.get("files")):
            if isinstance(file, dict) and isinstance(file.get("name"), str) \
                    and file["name"].lower().endswith(_MODEL_EXTENSIONS):
                yield file, version


def _search_candidates(items, match="family"):
    out = []
    for model in items:
        model = _dict(model)
        for file, version in _model_files(model):
            out.append(_candidate(file, version, model.get("name"), model.get("id"),
                                  model.get("nsfw"), match, model_type=model.get("type")))
    return out


def _folder_allows(folder, model_type):
    """False only when both the row folder and the CivitAI model type are known and disagree."""
    allowed = _FOLDER_TYPES.get((folder or "").lower())
    return not allowed or not model_type or model_type.lower() in allowed


def _family(items, exclude, model_id, base, exclude_name=None, folder=None):
    """Alternatives: same-model files first, then same-base files of other models."""
    pool = [c for c in _search_candidates(items) if _folder_allows(folder, c["model_type"])]
    if model_id is None and items:
        model_id = _dict(items[0]).get("id")
    same = [c for c in pool if c["_model_id"] == model_id]
    others = [c for c in pool if base and c["_model_id"] != model_id and c["base_model"] == base]
    out, seen = [], {exclude_name} if exclude_name else set()
    for c in same + others:
        if (exclude and c["sha256"] == exclude) or c["filename"] in seen:
            continue
        seen.add(c["filename"])
        out.append(_public(c))
        if len(out) >= _MAX_FAMILY:
            break
    return out


def _by_hash_candidate(data, sha):
    data = _dict(data)
    model = _dict(data.get("model"))
    for file in _list(data.get("files")):
        if isinstance(file, dict) and isinstance(file.get("name"), str) and _sha(file) == sha:
            return _candidate(file, data, model.get("name"), data.get("modelId"), model.get("nsfw"), "hash",
                              model_type=model.get("type"))
    return None


def _usable_mirrors(data):
    """Live, free archive mirrors; non-Hugging Face first (the HF path skips our sha256 check)."""
    mirrors = [m for m in map(_dict, _list(_dict(data).get("files")))
               if m.get("deletedAt") is None and not m.get("is_gated") and not m.get("is_paid")
               and isinstance(m.get("url"), str) and m["url"] and isinstance(m.get("filename"), str)]
    return sorted(mirrors, key=lambda m: _is_hf_url(m["url"]))


def _mirror_candidate(data, sha, base):
    model = _dict(_dict(data).get("model"))
    mirrors = _usable_mirrors(data)
    if not mirrors:
        return None
    mirror = mirrors[0]
    return {
        "filename": mirror["filename"],
        "url": mirror["url"],
        "sha256": sha,
        "size_bytes": None,
        "base_model": base,
        "version_name": "",
        "model_name": model.get("name") or "",
        "nsfw": bool(model.get("is_nsfw")),
        "site": "civitai_archive",
        "match": "hash",
        "civitai_type": None,
        "model_type": model.get("type") if isinstance(model.get("type"), str) else None,
        "_model_id": model.get("id"),
    }


def lookup(filename, budget, status_cb=None, folder=None):
    """Find ``filename`` (or same-family alternatives) on CivitAI / its archive.

    Returns ``{"exact": Candidate | None, "family": [Candidate]}`` or None when
    nothing was found. A 429 exhausts the shared ``budget``; whatever was found
    before the lookup stopped is still returned. A known ``folder`` (top model
    folder of the row) drops family files of a CivitAI type that does not
    belong there; the exact match is never filtered.
    """
    return _lookup(_basename(filename), budget, status_cb, folder)


def _fetch(url, budget, status_cb, source, auth=False):
    if not budget.spend():
        raise _Stop()
    if status_cb:
        status_cb(source)
    return _get_json(url, auth)


def _top_model_base(items):
    """Base model of the top search hit when all its versions agree."""
    versions = [_dict(v) for v in _list(_dict(items[0]).get("modelVersions"))] if items else []
    bases = {v.get("baseModel") for v in versions}
    return bases.pop() if len(bases) == 1 and None not in bases else None


def _lookup(wanted, budget, status_cb, folder=None):
    state = {"items": [], "exact": None, "base": None}
    try:
        _resolve(wanted, budget, status_cb, state)
    except RateLimited:
        budget.exhaust()
    except _Stop:
        pass
    items, exact = state["items"], state["exact"]
    base = (exact["base_model"] if exact else None) or state["base"] or _top_model_base(items)
    family = _family(items, exact["sha256"] if exact else None,
                     exact["_model_id"] if exact else None, base,
                     exact["filename"] if exact else None, folder)
    if not exact and not family:
        return None
    return {"exact": _public(exact) if exact else None, "family": family}


def _resolve(wanted, budget, status_cb, state):
    wanted_l = wanted.lower()
    generic = _is_generic_name(wanted)
    seen_ids = set()
    state["items"] = []
    for query in search_queries(wanted):
        url = f"{CIVITAI_API}/api/v1/models?query={urllib.parse.quote(query)}&limit=5&nsfw=true"
        found = [i for i in _list(_dict(_fetch(url, budget, status_cb, "civitai_search", True)).get("items"))
                 if isinstance(i, dict) and i.get("id") not in seen_ids]
        seen_ids.update(i.get("id") for i in found)
        state["items"].extend(found)
        # A generic name can't be proven by name, so the first hit is enough;
        # otherwise keep trying the next query until one holds the exact file.
        if found and (generic or any(c["filename"].lower() == wanted_l
                                     for c in _search_candidates(found, "exact"))):
            break

    pool = _search_candidates(state["items"], "exact")
    if not generic:
        state["exact"] = next((c for c in pool if c["filename"].lower() == wanted_l), None)
        if state["exact"]:
            return

    sha, state["base"] = _archive_hash(wanted, budget, status_cb)
    if not sha:
        return
    if generic:  # proof = a search file with this name whose sha256 the archive confirms
        pool = [c for c in pool if c["filename"].lower() == wanted_l]
    exact = next((c for c in pool if c["sha256"] == sha), None)
    if exact:
        exact["match"] = "hash"
    elif generic:
        # The archive hash is only a name hit too; without a search file that
        # shares it there is no proof, so leave the name matches as family.
        return
    else:
        data = _fetch(f"{CIVITAI_API}/api/v1/model-versions/by-hash/{sha}", budget, status_cb, "civitai_hash", True)
        exact = _by_hash_candidate(data, sha)
    if not exact:
        data = _fetch(f"{ARCHIVE_API}/api/sha256/{sha}", budget, status_cb, "civitai_archive")
        exact = _mirror_candidate(data, sha, state["base"])
    state["exact"] = exact


def _archive_hash(wanted, budget, status_cb):
    """Archive search: (sha256, base model) of the result named ``wanted``."""
    url = f"{ARCHIVE_API}/api/search?q={urllib.parse.quote(wanted)}"
    results = [r for r in _list(_dict(_fetch(url, budget, status_cb, "civitai_archive")).get("results"))
               if isinstance(r, dict)]
    for r in results:
        path = r.get("url")
        if str(r.get("name", "")).lower() == wanted.lower() and isinstance(path, str) \
                and path.startswith("/sha256/"):
            return path[len("/sha256/"):].lower(), r.get("base_model") or None
    return None, None


def _is_generic_name(filename):
    return os.path.splitext(_basename(filename))[0].lower() in _GENERIC_STEMS


def _is_hf_url(url):
    return _host(url or "") in ("huggingface.co", "www.huggingface.co") or _host(url or "").endswith(".huggingface.co")


def _default_hf_sha256(repo, path):
    from .downloader import get_remote_file_metadata, get_token
    return get_remote_file_metadata(repo, path, token=get_token())[1]


def _format_size(size):
    if not size:
        return ""
    if size < 1024 ** 3:
        return f"{max(1, round(size / 1024 ** 2))} MB"
    return f"{size / 1024 ** 3:.1f} GB"


def _label(candidate, relation=None):
    first = "CivitAI Archive" if candidate.get("site") == "civitai_archive" else "CivitAI"
    name = " ".join(p for p in (candidate.get("model_name"), candidate.get("version_name")) if p)
    parts = [first, name, candidate.get("filename") or "",
             _format_size(candidate.get("size_bytes")),
             candidate.get("base_model") or "",
             "NSFW" if candidate.get("nsfw") else "",
             relation or ""]
    return " · ".join(p for p in parts if p)


def _alternative(candidate, folder, relation=None):
    alt = {
        "filename": candidate["filename"],
        "url": candidate["url"],
        "sha256": candidate.get("sha256"),
        "source": candidate.get("site", "civitai"),
        "label": _label(candidate, relation),
    }
    if folder:
        alt["suggested_folder"] = folder
    return alt


def _append_alternatives(row, alts):
    existing = row.get("alternatives") if isinstance(row.get("alternatives"), list) else []
    urls = {a.get("url") for a in existing if isinstance(a, dict)}
    row["alternatives"] = existing + [a for a in alts if a["url"] not in urls]


def _same_file_relation(row, exact, hf_sha256, budget):
    if not exact.get("sha256") or not row.get("hf_repo") or not row.get("hf_path"):
        return None
    if not budget.spend():  # the HF metadata call counts against the scan budget
        return None
    try:
        hf_sha = hf_sha256(row["hf_repo"], row["hf_path"])
    except Exception:
        return None
    if not hf_sha:
        return None
    return "same file" if hf_sha.lower() == exact["sha256"] else "different file"


def _apply_result(row, result, hf_sha256, budget):
    exact, family = result.get("exact"), result.get("family") or []
    folder = row.get("suggested_folder")
    alts = []
    if row.get("url"):
        # HF row: CivitAI only fills the both-sources picker with the same model.
        if exact and exact.get("url"):
            alts.append(_alternative(exact, folder, _same_file_relation(row, exact, hf_sha256, budget)))
    else:
        if exact and exact.get("url"):
            row["url"] = exact["url"]
            row["sha256"] = exact.get("sha256")
            row["source"] = exact.get("site", "civitai")
        alts.extend(_alternative(c, folder) for c in family if c.get("url"))
    if alts:
        _append_alternatives(row, alts)


def _row_folder(row):
    folder = row.get("suggested_folder") or row.get("directory")
    if not isinstance(folder, str):
        return None
    top = re.split(r"[\\/]", folder.strip().strip("/\\"))[0]
    return top or None


def _cancel_or_skip(md, request_id, name):
    """Raise on cancel; True when the user skipped ``name``."""
    if not request_id:
        return False
    if md.is_search_cancelled(request_id):
        raise md.SearchCancelledException("Search cancelled by user")
    return md.is_model_skipped(request_id, name)


def search_missing(models, *, skip_all, skip_filenames, request_id, status_cb=None, hf_sha256=None):
    """Offer CivitAI matches for missing ``models`` (rows are mutated in place).

    Unresolved rows get the exact match as their URL; rows with a Hugging Face
    URL keep it and get CivitAI as an alternative. Other URLs are untouched.
    """
    from .downloader import civitai_search_enabled
    if skip_all or not civitai_search_enabled():
        return
    from . import model_discovery as md  # lazy: model_discovery imports this module
    hf_sha256 = hf_sha256 or _default_hf_sha256
    budget = Budget()
    _reset_auth_warning()
    # Unresolved rows first: HF rows only get an optional alternative, so they
    # use whatever budget is left.
    rows = sorted(models, key=lambda r: bool(r.get("url")))
    for row in rows:
        name = row.get("filename") or ""
        url = row.get("url")
        if not name or (url and not _is_hf_url(url)):
            continue
        if name.lower() in skip_filenames or _cancel_or_skip(md, request_id, name):
            continue
        if budget.remaining <= 0:
            break
        if request_id:
            md.set_current_searching_model(request_id, name)

        def report(source, name=name):
            if status_cb:
                status_cb({"message": f"Searching CivitAI for {name}", "source": source, "filename": name})

        try:
            result = lookup(name, budget, report, folder=_row_folder(row))
        finally:
            if request_id:
                md.set_current_searching_model(request_id, None)
        if result and not _cancel_or_skip(md, request_id, name):
            _apply_result(row, result, hf_sha256, budget)
