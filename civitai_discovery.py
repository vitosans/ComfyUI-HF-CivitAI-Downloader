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

_MAX_QUERIES = 3
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
    """Return up to 3 search strings for ``filename``, most specific first."""
    base = re.split(r"[\\/]", filename or "")[-1]
    stem = os.path.splitext(base)[0]
    first, *rest = stem.split("_")

    words = []
    for piece in _SEPARATORS.split(first):
        words.extend(_words(piece))
    for segment in rest:
        if _NOISE_TOKEN.match(segment):
            continue
        for piece in _SEPARATORS.split(segment):
            if piece and not _NOISE_TOKEN.match(piece):
                words.extend(_words(piece))

    queries = []
    for count in (len(words), 4, 2):
        query = " ".join(words[:count])
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


def _candidate(file, version, model_name, model_id, nsfw, match, site="civitai", url=None):
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
                                  model.get("nsfw"), match))
    return out


def _family(items, exclude, model_id, base, exclude_name=None):
    """Alternatives: same-model files first, then same-base files of other models."""
    pool = _search_candidates(items)
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
            return _candidate(file, data, model.get("name"), data.get("modelId"), model.get("nsfw"), "hash")
    return None


def _mirror_candidate(data, sha, base):
    data = _dict(data)
    model = _dict(data.get("model"))
    for mirror in _list(data.get("files")):
        mirror = _dict(mirror)
        if mirror.get("deletedAt") is None and not mirror.get("is_gated") and not mirror.get("is_paid") \
                and mirror.get("url") and isinstance(mirror.get("filename"), str):
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
                "_model_id": model.get("id"),
            }
    return None


def lookup(filename, budget, status_cb=None):
    """Find ``filename`` (or same-family alternatives) on CivitAI / its archive.

    Returns ``{"exact": Candidate | None, "family": [Candidate]}`` or None when
    nothing was found. A 429 exhausts the shared ``budget``; whatever was found
    before the lookup stopped is still returned.
    """
    return _lookup(_basename(filename), budget, status_cb)


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


def _lookup(wanted, budget, status_cb):
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
                     exact["filename"] if exact else None)
    if not exact and not family:
        return None
    return {"exact": _public(exact) if exact else None, "family": family}


def _resolve(wanted, budget, status_cb, state):
    wanted_l = wanted.lower()
    for query in search_queries(wanted):
        url = f"{CIVITAI_API}/api/v1/models?query={urllib.parse.quote(query)}&limit=5&nsfw=true"
        state["items"] = [i for i in _list(_dict(_fetch(url, budget, status_cb, "civitai_search", True)).get("items"))
                          if isinstance(i, dict)]
        if state["items"]:
            break

    pool = _search_candidates(state["items"], "exact")
    exact = next((c for c in pool if c["filename"].lower() == wanted_l), None)
    state["exact"] = exact
    if exact:
        return

    sha, state["base"] = _archive_hash(wanted, budget, status_cb)
    if not sha:
        return
    exact = next((c for c in pool if c["sha256"] == sha), None)
    if exact:
        exact["match"] = "hash"
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
