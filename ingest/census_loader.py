"""
Census / historical demographics loader.

The 1901/1911 Irish Census has no public API or bulk download.
This module provides the best available free alternatives:

  1. CSO historical county-level statistics (data.cso.ie open data)
  2. Wikipedia demographic sections for Irish counties and townlands
     (scraped via the free MediaWiki API)

TODO: If the National Archives ever releases a bulk export, replace
      _load_cso_data() with a proper structured loader.
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.documents import Document
from config import WIKIPEDIA_API_BASE

# CSO (Central Statistics Office) open data API
# Historical tables available at: https://data.cso.ie
CSO_API_BASE = "https://data.cso.ie/api"

# Wikipedia articles that contain reliable 1901/1911 demographic info
CENSUS_WIKI_ARTICLES = [
    "Census in Ireland",
    "Demographics of the Republic of Ireland",
    "Irish people",
    "Population history of Ireland",
    "Gaeltacht",
    "Irish language",
    "Great Famine (Ireland)",
    "Irish diaspora",
]


class CensusLoader:
    """
    Load historical demographic context from free public sources.

    Since the 1901/1911 census has no API, this combines:
    - Wikipedia articles on Irish demographics and history
    - CSO open data for aggregate statistics
    """

    def __init__(self):
        self.client = httpx.Client(timeout=30)

    def load_all(self) -> list[Document]:
        """Load all available demographic context documents."""
        docs = []
        docs.extend(self._load_wikipedia_demographics())
        docs.extend(self._load_cso_data())
        return docs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _load_wikipedia_demographics(self) -> list[Document]:
        """Fetch Wikipedia articles about Irish demographics."""
        params = {
            "action": "query",
            "titles": "|".join(CENSUS_WIKI_ARTICLES),
            "prop": "extracts",
            "explaintext": True,
            "format": "json",
        }
        response = self.client.get(WIKIPEDIA_API_BASE, params=params)
        response.raise_for_status()
        data = response.json()

        docs = []
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if page.get("missing") is not None:
                continue
            title = page.get("title", "")
            extract = (page.get("extract") or "").strip()
            if extract and len(extract) > 100:
                docs.append(Document(
                    page_content=f"{title}\n\n{extract}",
                    metadata={
                        "source": "census_wikipedia",
                        "title": title,
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    },
                ))
        return docs

    def _load_cso_data(self) -> list[Document]:
        """
        Attempt to load aggregate statistics from CSO open data.
        Returns empty list gracefully if the endpoint is unavailable.

        CSO open data portal: https://data.cso.ie
        Historical census tables are available but endpoint stability varies.
        """
        try:
            # Population by county from historical censuses
            url = f"{CSO_API_BASE}/1.0/dataset/VSA08/data?format=JSON"
            response = self.client.get(url, timeout=15)
            if response.status_code != 200:
                return []

            data = response.json()
            text = self._cso_json_to_text(data, "Historical Irish Population by County")
            if text:
                return [Document(
                    page_content=text,
                    metadata={"source": "cso", "dataset": "VSA08", "url": url},
                )]
        except Exception:
            pass

        return []

    def _cso_json_to_text(self, data: dict, title: str) -> str:
        """Convert a CSO JSON-stat dataset to readable text."""
        try:
            dims = data.get("dimension", {})
            values = data.get("value", [])
            if not dims or not values:
                return ""
            lines = [title]
            # Simple flattening — CSO JSON-stat can be complex
            for k, v in dims.items():
                cats = v.get("category", {}).get("label", {})
                if cats:
                    lines.append(f"{k}: {', '.join(list(cats.values())[:10])}")
            return "\n".join(lines)
        except Exception:
            return ""
