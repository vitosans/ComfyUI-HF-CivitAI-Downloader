"""CivitAI lookup for models that were not found on Hugging Face (stdlib only)."""
import os
import re

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
