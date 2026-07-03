"""Pluggable tool backends for SPORK's web-search benchmarks (GAIA, HotpotQA).

The SPORK paper evaluated GAIA and HotpotQA against internal web-search and
web-browse services that are not part of this public release. To run those
benchmarks you must supply your own backend by subclassing the abstract
placeholders below and wiring them into the corresponding tool executor in
``spork_unified_eval.py``.

Both classes are intentionally dependency-free (stdlib only). A real
implementation might wrap SerpAPI, Bing, a local Wikipedia dump, or any other
search/index of your choosing.

Example::

    from tool_backends import WebSearchBackend

    class MySearch(WebSearchBackend):
        def search(self, query, top_k=8):
            hits = my_search_api(query, n=top_k)
            return [
                {"title": h.title, "url": h.url, "snippet": h.summary}
                for h in hits
            ]
"""
from __future__ import annotations


class WebSearchBackend:
    """Abstract web-search backend.

    Subclass and implement :meth:`search`. The SPORK eval calls it with the
    model-issued query string; the returned hits are formatted into the tool
    result text shown to the model.
    """

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        """Run a web search and return up to ``top_k`` results.

        Args:
            query: The search query issued by the model.
            top_k: Maximum number of results to return.

        Returns:
            A list of result dicts, each with the keys::

                {
                    "title":   str,   # result title / page name
                    "url":     str,   # canonical URL of the result
                    "snippet": str,   # short text excerpt / summary
                }

            An empty list signals "no results found".
        """
        raise NotImplementedError(
            "Plug in your own web-search API (e.g. SerpAPI/Bing/a local index). "
            "The paper used internal APIs not included in this release."
        )


class WebBrowseBackend:
    """Abstract web-browse backend.

    Subclass and implement :meth:`browse`. The SPORK eval calls it with a URL
    (and an optional natural-language goal) and shows the returned page text to
    the model.
    """

    def browse(self, url: str, goal: str = "") -> str:
        """Fetch a URL and return its readable page text.

        Args:
            url: The URL the model asked to visit.
            goal: Optional natural-language description of what the model is
                looking for; a backend may use it to focus extraction, or
                ignore it.

        Returns:
            The page text content as a string. An empty string signals that no
            content could be extracted.
        """
        raise NotImplementedError(
            "Plug in your own web-browse/fetch API (e.g. a headless browser or "
            "an HTML-to-text extractor). The paper used internal APIs not "
            "included in this release."
        )


class WikipediaSearchBackend(WebSearchBackend):
    """Free, key-less Wikipedia search backend (public MediaWiki API).

    HotpotQA is Wikipedia-based QA, so this provides an out-of-the-box,
    fully-reproducible HotpotQA run with no API key. It searches for matching
    article titles, then fetches each article's intro extract as the snippet.

    stdlib-only (urllib). Honors the ``http_proxy`` / ``https_proxy`` environment
    variables if set (urllib uses them automatically).
    """

    API = "https://en.wikipedia.org/w/api.php"

    # Descriptive UA per Wikipedia API etiquette (generic UAs get throttled harder).
    UA = "SPORK-HotpotQA/1.0 (https://github.com/; research eval; one request at a time)"

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        import json, time, urllib.parse, urllib.request, urllib.error

        def _get(params, retries=4):
            url = self.API + "?" + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={"User-Agent": self.UA})
            for attempt in range(retries):
                try:
                    with urllib.request.urlopen(req, timeout=25) as r:
                        return json.load(r)
                except urllib.error.HTTPError as e:
                    if e.code in (403, 429, 503) and attempt < retries - 1:
                        time.sleep(1.5 * (2 ** attempt))  # backoff on rate-limit
                        continue
                    raise
            return {}

        if not query:
            return []
        hits = _get({"action": "query", "list": "search", "srsearch": query,
                     "srlimit": top_k, "format": "json"}).get("query", {}).get("search", [])
        titles = [h["title"] for h in hits]
        if not titles:
            return []
        pages = _get({"action": "query", "prop": "extracts", "exintro": 1,
                      "explaintext": 1, "redirects": 1, "titles": "|".join(titles),
                      "format": "json"}).get("query", {}).get("pages", {})
        extract_by_title = {p.get("title", ""): p.get("extract", "") for p in pages.values()}
        out = []
        for t in titles:
            out.append({
                "title": t,
                "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(t.replace(" ", "_")),
                "snippet": (extract_by_title.get(t, "") or "")[:1500],
            })
        return out
