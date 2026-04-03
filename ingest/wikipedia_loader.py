"""
Wikipedia Irish places loader.

Uses the free MediaWiki API (no key required) to fetch articles about
Irish townlands, parishes, counties, and geographical features.
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.documents import Document
from config import WIKIPEDIA_API_BASE

# Seed categories covering Irish places well
IRISH_PLACE_CATEGORIES = [
    "Townlands of County Clare",
    "Townlands of County Cork",
    "Townlands of County Donegal",
    "Townlands of County Galway",
    "Townlands of County Kerry",
    "Townlands of County Mayo",
    "Townlands of County Sligo",
    "Townlands of County Tipperary",
    "Civil parishes of Ireland",
    "Islands of Ireland",
    "Mountains of Ireland",
    "Loughs of Ireland",
]


class WikipediaIrishPlacesLoader:
    """Load Irish place articles from Wikipedia using the free MediaWiki API."""

    def __init__(self):
        self.client = httpx.Client(timeout=30)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _get(self, params: dict) -> dict:
        params.setdefault("format", "json")
        params.setdefault("action", "query")
        response = self.client.get(WIKIPEDIA_API_BASE, params=params)
        response.raise_for_status()
        return response.json()

    def search(self, query: str, max_results: int = 5) -> list[Document]:
        """Search Wikipedia for Irish place articles matching a query."""
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": f"{query} Ireland",
            "srlimit": max_results,
            "srprop": "snippet|titlesnippet",
        }
        data = self._get(search_params)
        results = data.get("query", {}).get("search", [])
        titles = [r["title"] for r in results]
        return self._fetch_articles(titles)

    def load_by_title(self, title: str) -> Document | None:
        """Fetch a single Wikipedia article by exact title."""
        docs = self._fetch_articles([title])
        return docs[0] if docs else None

    def load_category(self, category: str, max_pages: int = 100) -> list[Document]:
        """Fetch all articles in a Wikipedia category."""
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": max_pages,
            "cmtype": "page",
        }
        data = self._get(params)
        members = data.get("query", {}).get("categorymembers", [])
        titles = [m["title"] for m in members]
        return self._fetch_articles(titles)

    def load_irish_counties(self) -> list[Document]:
        """Fetch overview articles for all 32 Irish counties."""
        counties = [
            "County Antrim", "County Armagh", "County Carlow", "County Cavan",
            "County Clare", "County Cork", "County Derry", "County Donegal",
            "County Down", "County Dublin", "County Fermanagh", "County Galway",
            "County Kerry", "County Kildare", "County Kilkenny", "County Laois",
            "County Leitrim", "County Limerick", "County Longford", "County Louth",
            "County Mayo", "County Meath", "County Monaghan", "County Offaly",
            "County Roscommon", "County Sligo", "County Tipperary", "County Tyrone",
            "County Waterford", "County Westmeath", "County Wexford", "County Wicklow",
        ]
        return self._fetch_articles(counties)

    def _fetch_articles(self, titles: list[str]) -> list[Document]:
        """Batch fetch article extracts for a list of titles."""
        if not titles:
            return []

        # Wikipedia API allows up to 50 titles per request
        docs = []
        for i in range(0, len(titles), 50):
            batch = titles[i : i + 50]
            params = {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "extracts|categories|coordinates",
                "exintro": True,       # intro section only (keeps chunks focused)
                "explaintext": True,   # plain text, no HTML
                "cllimit": 5,
                "coprop": "type",
            }
            data = self._get(params)
            pages = data.get("query", {}).get("pages", {})

            for page in pages.values():
                doc = self._page_to_document(page)
                if doc:
                    docs.append(doc)

        return docs

    def _page_to_document(self, page: dict) -> Document | None:
        """Convert a MediaWiki page dict to a LangChain Document."""
        if page.get("missing") is not None:
            return None

        title = page.get("title", "")
        extract = (page.get("extract") or "").strip()
        if not extract or len(extract) < 50:
            return None

        # Coordinates if available
        coords = page.get("coordinates", [])
        lat = coords[0].get("lat") if coords else None
        lon = coords[0].get("lon") if coords else None

        # Category labels
        cats = [c["title"].replace("Category:", "") for c in page.get("categories", [])]

        metadata = {
            "source": "wikipedia",
            "title": title,
            "page_id": page.get("pageid", ""),
            "lat": lat,
            "lon": lon,
            "categories": ", ".join(cats[:5]),
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        }

        return Document(page_content=f"{title}\n\n{extract}", metadata=metadata)
